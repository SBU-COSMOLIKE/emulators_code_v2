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

## Scheduler execution protocol (45M-25, Architect-VERIFIED, open)

`make_scheduler` advertises a constructible scheduler class, but constructing
an object does not establish how often or with which argument it must be
stepped. The training loop implements only two per-epoch protocols:
`ReduceLROnPlateau.step(validation_median)` and a bare `scheduler.step()` for
every other class. A per-update scheduler such as `OneCycleLR` therefore
constructs successfully and then executes at the wrong cadence by orders of
magnitude. Per-phase and refinement construction also need the true number of
optimizer updates rather than the nominal epoch count.

Required contract:

1. Either expose a deliberately bounded scheduler surface and reject every
   unsupported class before model/data staging, or persist an explicit
   protocol with each class: cadence (`per_update` or `per_epoch`), whether a
   metric is required, which metric, and the resolved horizon.
2. A per-update scheduler steps only after an accepted optimizer update. A
   nonfinite/skipped update or an empty chunk does not advance it.
3. A per-epoch scheduler steps once after the complete accepted epoch.
   `ReduceLROnPlateau` uses the shared validated ordinary-median reducer and
   chi-squared domain rule on the raw-model scores. With EMA disabled this is
   also the reported/selection median; with EMA active, the deliberate policy
   remains raw-model median for scheduling and EMA-model median for reporting
   and selection.
4. Each trunk, head, joint, and refinement pass resolves its own effective
   update count from executed rows, batch size, chunk-tail policy, and epoch
   count. Nominal configuration counts cannot stand in for executed steps.
5. Warmup has one owner and cannot advance both the explicit warmup rule and a
   scheduler's internal warmup on the same update.
6. The resolved pass record stores scheduler class, kwargs, cadence, metric
   source, and effective step count so the artifact describes what actually
   ran.

The board gate uses counting schedulers for per-update and per-epoch paths,
pins the plateau median argument, and checks a short analytic learning-rate
sequence for an admitted `OneCycleLR` or the exact startup refusal if that
class is outside the bounded surface. It covers a ragged final chunk, a
skipped optimizer step, separate phase horizons, refinement, and a
double-warmup mutation. Merely printing the scheduler class or completing a
run does not prove the execution protocol.

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

### 25M-04 amendment (Red Team CONFIRMED, awaiting Architect adjudication): the direct cosmic-shear driver no longer selects its historic study name

The family-identity repair changed the public `main` default from
`family=None` to `family="cosmolike"` (commit `e9943bc`) but left the earlier
selection expression unchanged:
`study_name = STUDY_NAME if family is None else prog`
(`cosmic_shear_tune_emulator.py:217,287-290`). Under the public defaults, the
constant `STUDY_NAME="cosmic_shear_tune"` is unreachable and the command
selects `cosmic_shear_tune_emulator`. The adjacent comment still promises
that the cosmic-shear study keeps its historic name. An AST probe of the real
signature and assignment gives exactly this mismatch.

Concrete wrong result: before `e9943bc`, the documented direct command and
default journal used the study `cosmic_shear_tune`; after the family validator
fix, the same command, file, and scientific inputs open/create the different
study `cosmic_shear_tune_emulator`. Optuna identity is `(journal path,
study_name)`, so the advertised resume silently forks. Per-family wrappers
legitimately use their pinned `prog`; the regression is the direct
cosmic-shear special case. This amends unit 53: the manifest needs a stable,
canonical family study name as well as scientific content identity.

Required contract: one pure resolver maps explicit family identity to a
stable study name. The direct `cosmolike` family maps to the retained historic
constant; each thin family wrapper maps to its stable family tag. The resolved
name is part of the study manifest and final report. Any intended rename is an
explicit migration/refusal, never collateral of a validator change.

CPU red legs: signature-default resolution returns the historic cosmic-shear
name; every wrapper returns its unique stable name; existing-journal lookup
resumes the expected study rather than creating a sibling; manifest and
Optuna names agree; and a mutation restoring the `family is None` conditional
reproduces the fork and must red.

## 25M-05 (Red Team CONFIRMED, awaiting Architect adjudication): science-curve headers record the raw activation flag instead of the activation that ran

Both sweep engines resolve activation through
`EmulatorExperiment.from_config`: when `--activation` is absent, the family
comes from `train_args.model.activation`, then defaults to `H`
(`experiment.py:2181-2189` and the parallel family branches). The resolved
value is stored in `exp.activation` and printed in the design banner. The
result writers do not use it. The N-train sweep writes
`"activation": args.activation`
(`cosmic_shear_sweep_ntrain_emulator.py:413,490-496`); the ordinary
hyperparameter sweep writes the same raw flag unless activation itself is the
sweep axis (`cosmic_shear_sweep_hyperparam_emulator.py:273,441-447`).

Concrete wrong result: with the shipped YAML/default path and no CLI flag,
the trained model uses activation `H`, while the public table header is
`# activation=None`. The dependency-free real `save_learning_curves` body was
executed with the driver's actual metadata value and reproduced that line. A
YAML selecting another family is mislabeled the same way. A reader cannot
reconstruct which design produced the science curve even though the process
already owns the resolved fact. Unit 41 owns resolved-run truth; this extends
that doctrine to sweep products rather than creating another resolver.

Required contract: metadata is assembled from one immutable resolved run
record, never raw optional CLI fields. N-train and non-activation
hyperparameter sweeps persist the actual shared activation and any head pin;
an activation-family sweep records `swept` plus ordered values. Serial and
worker paths use the same record, and banner, artifact, table, and figure
labels agree. Raw CLI provenance may be separate but cannot masquerade as
executed state.

Red legs: YAML/default `H` with no flag records `H`; a YAML `power` selection
records `power`; an explicit CLI override records the override; an activation
sweep records `swept` plus values; serial and pooled metadata agree; table
readback equals the experiment's resolved value; and a mutation using
`args.activation` recreates `activation=None` and must red. Table assembly is
pure CPU; one real sweep-path integration leg may be board-listed for Vivian's
workstation.

## 25M-10 (Red Team CONFIRMED, awaiting Architect adjudication): `--quiet` cannot satisfy its public all-stdout contract

Every train/sweep/tune help surface promises that `--quiet` suppresses all
stdout. Experiments pass `verbose=not self.quiet` into `load_source`, but
`load_source` calls `stage_source`, whose resource line is an unconditional
raw `print` and whose signature has no verbosity/emit channel
(`data_staging.py:221,306,560-750`). An AST-extracted execution of the real
function under redirected stdout produced the staging line even though the
outer caller has no way to disable it. CMB geometry adds three more raw prints
(`experiment.py:3493,3504,3548`), and parallel N-train/hyperparameter worker
failure paths print directly (`cosmic_shear_sweep_ntrain_emulator.py:166`,
`cosmic_shear_sweep_hyperparam_emulator.py:175`) rather than respecting the
driver's quiet logger.

Concrete public result: a valid `--quiet` run with ordinary staged data emits
at least `stage_source: ...`; a CMB run also emits its staging summary. On a
worker failure the nominally quiet sweep emits child-process stdout. This is
not cosmetic under the repository's documentation rule: printed lines are
the machine/operator record, and the CLI advertises a deterministic empty
stdout surface for batch composition.

Required contract: one explicit output channel is threaded through every
reachable staging, geometry, training, and worker path. Under `--quiet`,
successful stdout is empty; files still publish normally. Errors remain loud
through nonzero status and stderr (quiet must not swallow failure). Essential
scientific facts belong in the persisted resolved record and ordinary banner,
not as raw-print exemptions. Remove raw `print` calls from methods reached by
quiet drivers or inject the owner logger; do not use process-global stdout
redirection as hidden state.

CPU legs capture the real public paths for a valid plain/scalar/CMB staging
run and require empty stdout under quiet plus nonempty current output without
quiet; a worker failure is nonzero with diagnostic stderr and empty stdout;
serial and spawned behavior agree; and an AST/call census proves every
reachable output site uses the owner channel. The mutation restoring
`stage_source`'s raw print must reproduce the captured line and red.

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

## UNIT 9 AMENDED (45M-51): the scalar driver re-declares diagnostic eligibility from its family name

Seventh 45M batch (2026-07-12), Architect-verified on HEAD; joins
this note's generic-diagnostics section (unit 9) beside the
thirteenth-wave extension. Scalar NPCE is explicitly legal
(validate_scalar permits the top-level pce: block; every PCE loss
declares needs_params = True, losses/pce.py :79/:216/:386), yet
scalar_train_emulator.py --diagnostic asserts in prose that "the
scalar loss is a plain chi2, so the local-linear floor applies too"
(:251-252) and calls local_linear_floor UNCONDITIONALLY (:267);
local_linear_floor refuses any needs_params loss with a ValueError
(diagnostics.py:185-188). The save happens BEFORE the diagnostic
(:212 vs :237), so every valid scalar NPCE diagnostic command
trains, saves the artifact, then raises instead of producing the
promised PDF — a deterministic failed command. The shared family
driver already contains the correct capability branch
(cosmic_shear_train_emulator.py:500: floor only when NOT
needs_params, else a truthful logged skip); the scalar fork omitted
that rule.

Contract: diagnostic eligibility is owned by the
diagnostic/capability layer, never re-declared by a driver from its
family name. One shared diagnostic orchestrator decides which
analyses run and emits a structured availability record (available,
or unavailable with a reason) consumed by both logging and plotting.
Scalar NPCE skips only the local-linear floor and still produces its
coverage, hardness, and scalar residual pages plus the PDF; plain
scalar behavior is unchanged.

Red legs: plain scalar diagnostic executes the floor; NPCE scalar
diagnostic marks the floor unavailable and COMPLETES the PDF;
mutation arm — restore the unconditional scalar call and the gate
must fail; the scalar and shared-family drivers produce identical
availability semantics for synthetic losses with needs_params
true/false; the "scalar loss is plain" prose disappears, replaced by
the actual capability rule. The integration leg (torch: real
predictor/loss path) joins the existing board-listed scalar-smoke or
diagnostics check under gates/checks/, workstation-run; the
eligibility helper itself gets a pure CPU no-torch unit leg.

## UNIT 14 REOPENED (45M-53 + addendum): a finite negative chi2 ranks better than perfect — increment (e)

Eighth 45M batch (2026-07-12), Architect-verified on HEAD (post
63880d1). The increment-(c) producer guard is TRAINING-ONLY:
_validate_chi2 runs once at the top of _reduce
(losses/core.py:408); the validation and diagnostic paths never
call _reduce — eval_val guards torch.isfinite only
(training.py:1490) and eval_source_chi2 likewise (:1572). A finite
negative per-sample chi2 therefore passes both: it enters mean and
median as-is (:1497-1498), compares FALSE to every positive
threshold in the (Nval, T) grid (:1504), so the corrupted row
counts as PERFECT, lowers frac>0.2, and can crown the corrupted
epoch in the best-epoch selection. eval_source_chi2 publishes the
negative as a diagnostic delta-chi2. Training rejects the same
producer that scoring accepts — the contract is internally
inconsistent, and this is a wrong SELECTION result, not a message
defect. Reachable today: the geometry-SPD work (units 11/13) is
queued, not landed, so a non-PSD precision reaches chi2 unchecked
(unit 11: from_state validates nothing), and increment (c)'s own
rationale concedes roundoff negatives in the rescaled/transfer
contractions. The one-output example: precision [-1], residual 2
-> chi2 = -4, finite.

Gate gap (the addendum, verified): the board advertises "eager and
torch.compile agree" (gates/board.py:1243-1248), but
finite_contract.py wraps the WHOLE compiled arm — construction,
execution, backward, assertion setup — in one broad
`except Exception` that reports a soft-skip (:703-712). On the CUDA
workstation an Inductor regression, device mismatch, or broken
backward becomes a green skip: the exact failure class the leg
exists to catch.

ARCHITECT RULINGS (the two Implementer flags + the red team's
tolerance escalation, adjudicated here so the committed constant
does not become the contract by default):

1. Fifth sqrt site: CONFIRMED. The increment-(c) contract is every
   sqrt site whose argument can be zero under valid input; the
   berhu_capped region-3 where-mask leak is squarely in contract.
   Approved as implemented, no rework.
2. Tolerance: the absolute _CHI2_NEG_TOL = 1e-6 is SUPERSEDED (the
   recorded requirement said scale-aware; the flagged deviation is
   adjudicated against the constant). The band becomes
   band = max(1.0e-6, _CHI2_NEG_KAPPA * eps(compute dtype) * n_terms)
   with _CHI2_NEG_KAPPA = 32 documented, and n_terms = the per-row
   count of summed products in the active contraction (n_dv for the
   plain whitened form; the documented equivalent for the
   rescaled/transfer forms). n_terms is a PER-RUN constant known at
   build time, so the band stays elementwise with no data-dependent
   branch (compile/CUDA-graph safe) and no batch statistic — which
   answers the Implementer's poisoning objection (a NaN cannot
   poison a constant) while being scale-aware in exactly the
   quantity roundoff grows with. The old 1e-6 survives only as the
   band's floor.

Contract, increment (e) — adopted from the handoff:

1. ONE shared chi2-domain predicate (finite first, then nonnegative
   within the band above) used by the training reduction, eval_val,
   and eval_source_chi2. The helper returns the normalized c and the
   bad mask; training folds bad entries to NaN (the landed per-step
   refusal, compile-safe); the eval/diagnostic boundaries RAISE
   before any median/mean/fraction computation and before any
   best-record comparison, naming side, count, first source rows,
   minimum value, and the allowed band.
2. Within-band negatives normalize to EXACT 0 through the same
   helper everywhere — training may never call a row exact while
   evaluation reports it negative.
3. The queued geometry-SPD unit does NOT substitute for this runtime
   defense (artifacts, contractions, and test doubles can all
   violate the producer invariant).
4. Every valid nonnegative score is preserved byte-for-byte.

Gate amendment (finite-contract, board-listed, torch, workstation):
real one-output r^T[-1]r = -4 driven through the REAL eval_val must
raise before fractions; the same through eval_source_chi2 must
raise; a negative through the compiled validation callable must
raise; a valid exact zero stays accepted; a positive control is
byte-identical; the finite-only predicate is the mutation arm and
must falsely crown the negative row (the best-epoch flip shown);
band-edge legs on BOTH sides of the adjudicated tolerance at a
documented n_terms (just inside -> exact 0; just outside ->
refusal). Compiled-leg truth (the addendum): capability detection is
explicit and runs BEFORE the test; on the workstation/CUDA lane,
compile construction + execution + backward are MANDATORY and any
exception is RED with the traceback; a genuinely compiler-less dev
box may emit only the board's explicit non-green skip status (the
SKIP-DEP class already exists in run_board.py) which can never count
toward closure — 33/33 on the workstation means zero skips; plus a
mutation control that forces the compiled callable to raise and
proves the gate cannot report green.

Unit 14 state: increments a+b+c+d landed (a0d03f5, 97963b8,
63880d1) and REOPENED for (e) + the gate truth amendment; (e) runs
FIRST in the work order, before 42+43 (same freshly-touched code,
one gate revision).

#### Finite-contract resume (2026-07-12, Opus) — increment (e) 45M-53 in; unit 14 closes on a+b+c+d+e+gate

Increment (e) implemented and self-committed on the branch (batch grant,
pending Architect audit). Both flags were ruled (fifth sqrt site confirmed;
the absolute tolerance superseded); this adopts the adjudicated band.

- ONE shared chi2-domain predicate (losses/core.py): `_chi2_domain(c, band)`
  returns (c_norm, bad) -- finite first, then nonnegative within the band;
  within-band roundoff negatives normalize to EXACT 0, `bad` marks
  non-finite OR materially-negative (< -band). Elementwise, no sync. The
  scale-aware band is `_chi2_neg_band(dtype, n_terms) = max(1e-6,
  _CHI2_NEG_KAPPA(32) * eps(dtype) * n_terms)`; `_chi2_n_terms()` is the
  per-row summed-product count -- w^2 on the DENSE base CosmolikeChi2
  (covers rescaled/transfer via inheritance), w on the DIAGONAL override
  CmbDiagonalChi2. The old absolute `_CHI2_NEG_TOL`/`_validate_chi2` are
  replaced.
- Three call sites, one predicate (training.py): `_reduce` folds bad -> NaN
  (the landed per-step refusal, compile-safe); `eval_val` and
  `eval_source_chi2` now RAISE (via `_report_chi2_domain`, naming side,
  count, first source rows, minimum value, and the band) before any
  mean/median/fraction or best-record comparison; within-band -> exact 0
  everywhere. A production loss always declares `_chi2_n_terms`; a bare test
  double defaults to the band floor (documented). Valid nonnegative scores
  byte-identical.
- Gate: Part H (finite_contract.py) -- eval_val -4 raises; eval_source_chi2
  -4 raises (side diagnostic); exact-zero accepted; all-positive control
  byte-identical; the finite-only false-crowning mutation (the -4 row lowers
  frac>0.2); band-edge both sides; a negative through the COMPILED reduce ->
  non-finite. The Part F compile arm is fixed per the addendum:
  `_can_compile()` capability detection FIRST, then the leg is MANDATORY-red
  on a compile-capable box (exception -> FAIL + traceback), with a
  raising-callable mutation control; a compiler-less box prints an explicit
  `[SKIP-DEP]` that never counts toward 33/33. Board maps + gate docstring
  name the 45M-53 clause.
- Mac gate (raw): py_compile OK (core.py, cmb.py, training.py,
  finite_contract.py, board.py); probe_chi2_domain.py 9/9 on the REAL
  `_chi2_neg_band` + `_chi2_domain` + `_safe_sqrt` -- the scale-aware band
  (floor + dense scaling), valid unchanged, within-band -> 0 (not bad),
  materially-negative (-4) and non-finite -> bad, band-edge both sides, the
  _safe_sqrt regression clean. The torch eval/compiled legs ride the
  workstation finite-contract gate (33/33, zero skips = compile mandatory).

Unit 14 closes on a + b (a0d03f5) + c (97963b8, 45M-24) + d (63880d1,
45M-47) + e (this, 45M-53) + the extended finite-contract gate (Parts A-H).
Files (e): emulator/losses/core.py, emulator/losses/cmb.py,
emulator/training.py, gates/checks/finite_contract.py, gates/board.py,
notes/training-stack.md. Next: 42 landed (5661c08); 43 proposed; then
50 -> 52 -> 55 -> 22 -> 13.

## UNIT 59 (45M-56, tenth batch): top-level config keys are never censused — a typo silently changes the training design

CONFIRMED (Fable, 2026-07-12). Every nested block is strictly
validated with an unknown-key census (param_cuts experiment.py:540,
per-head scheduler :282, data.cmb :686, data.grid :833, data.mps
:948), but NO top-level census exists anywhere: an untruncated grep
for set(cfg) / cfg.keys() / list(cfg) / sorted(cfg) across
experiment.py, the drivers, and gates/checks returns EMPTY. Branch
selection is pure cfg.get: transfer at :625/:772, pce at :757; the
sweep driver reads cfg["sweep"] raw (family_drivers.py:92, a bare
KeyError when absent). The red team drove the REAL extracted
validate_scalar: a config carrying trasnfer: {from: base, form: sum}
was accepted and resolved as an ordinary scalar run — consistent with
the census absence, since the only top-level readers are
cfg.get(<known>) calls. Consequences adopted verbatim: a requested
transfer silently becomes a from-scratch model; a requested NPCE run
becomes a plain emulator; a sweep block is ignored by a non-sweep
driver; and the RAW saved YAML claims a feature the resolved model
never executed — the never-trust-defaults inversion (the artifact
lies about the run).

Contract (the red team's six clauses, with three rulings):

1. One explicit top-level schema shared by config loading and every
   driver.
2. The allowed set covers the emulator-owned blocks (data,
   train_args, pce, transfer, sweep) plus the explicitly enumerated
   cobaya pass-through blocks the shared generator YAMLs need
   (params, likelihood, theory, and any deliberately supported
   controls). RULING (allowlist provenance): the set is not invented
   — the Implementer ENUMERATES every top-level key across the
   shipped corpus (example_yamls/, gates/configs/, the generator
   YAML docstrings), records the census in this note, and the
   Architect audits it before the schema hardens (propose-first; the
   large-unit precedent). Today the shipped training YAML carries
   only data + train_args; the generator YAMLs carry the cobaya
   blocks + train_args — both families must resolve under one
   schema.
3. Any other top-level key raises BEFORE device selection, staging,
   source loading, or artifact mutation — at from_config entry AND
   at each driver's load boundary.
4. RULING (suggestion mechanism): difflib.get_close_matches (stdlib,
   C-readable); the error names the unknown key, the close
   recognized spelling when one exists (trasnfer -> transfer, pec ->
   pce), and the full allowed set.
5. Driver-specific requirements stay driver-specific: the sweep
   driver requires sweep (its missing-sweep failure becomes a named
   error, not today's raw KeyError); the training driver may
   tolerate a VALID sweep block (one YAML deliberately serves both
   drivers) but never an unknown spelling.
6. The resolved record states the executed composition explicitly —
   plain, NPCE with form, fine-tune with source, or transfer with
   source/form/space; a raw block is never evidence that its path
   executed. INTERLOCK: this clause rides unit 41's resolved-record
   rebuild (artifacts-inference-warmstart.md) — one record, one
   writer.

Gate legs (pure CPU, board-listed with the config-schema coverage):
valid plain / PCE / transfer / sweep / shared-generator configs
pass; misspelled transfer, pce, sweep, train_args, data fail at the
top boundary naming the key and the suggestion; mutation arm
restores bare cfg.get("transfer") without the census and proves the
misspelled-transfer config reaches the plain branch; raw and
resolved records compared to prove the selected composition matches
execution.

Placement: campaign phase, beside unit 41 — the config namespace
truth companion to the resolved-record truth (NOT part of the
8+17+25+26 file-set authenticity cluster; that boundary is files,
this one is the config namespace above train_args).

## UNIT 60 (45M-57, tenth batch): the reported validation "median" is the lower middle sample for every even n_val

CONFIRMED (Fable, 2026-07-12) at HEAD 5661c08. eval_val reduces the
full validation chi2 vector with median = c.median().item()
(training.py:1498 — the single .median( site in the module; the line
number drifts with the in-flight 14(e) increment). torch.median
returns the LOWER of the two central ordered values for even length:
live-verified on the cocoa python —

    torch.median([0, 1, 9, 10])   = 1.0   (ordinary median: 5.0)
    torch.quantile([...], 0.5)    = 5.0
    torch.median([0, 1, 9])       = 1.0   (odd control; quantile
                                           agrees at 1.0)

The value is not cosmetic: sched_median = raw_median (:2147) feeds
scheduler.step(sched_median) (:2202) for ReduceLROnPlateau; the
best-epoch record breaks equal-frac ties on it (:2163-2166 — ties
are common because frac moves in steps of 1/n_val); the per-epoch
medians list is persisted and plotted as a scientific summary
(:2149, returned through :3056). And FIVE gate files manufacture
their reference statistics with the SAME lower-middle operation —
cmb_smoke.py:389, bsn_smoke.py:193, mps_smoke.py:203,
finite_contract.py:137/:873, ge_c_eval_bs.py:99 — so the board
ENCODES the defect rather than detecting it. Reachability: n_val is
any positive integer and even is the norm (shipped placeholder
n_val 5000; board runs use 200).

RULINGS:

- Naming: adopt the ordinary 50th-percentile median everywhere —
  the center value for odd N, the arithmetic mean of the two center
  values for even N. NOT renamed to "lower median": every prose,
  plot, history, and gate surface already says median and every
  scientific reader assumes the standard estimator; a rename sweep
  buys nothing.
- Implementation freedom: torch.quantile(c, 0.5) reproduces the
  contract on both parities (verified above; its documented
  input-size cap ~2^24 elements sits far above any n_val here and
  the helper documents it), or a kthvalue midpoint. Either way ONE
  shared reduction serves eval_val, the scheduler feed, the
  tie-break, saved histories, and every gate reference.
- The five gate reference sites migrate IN THIS UNIT, so repaired
  production code and the gates cannot disagree for the right
  reason.
- USER-VISIBLE, declared: even-n_val medians change (reported,
  persisted, plotted), and plateau timing / tie-breaks can change —
  training trajectories are not byte-identical for even n_val.
  Odd-n_val values are exactly unchanged (gate leg).
- Placement: unit 60 rides WITH unit 50 (epoch-truth under
  chunking) — the same eval/epoch-reduction surface, one visit —
  and lands AFTER 14(e) (same function; no collision with the
  in-flight increment).

Gate legs (torch, workstation lane, joining
gates/checks/ge_c_eval_bs.py — board-run at board.py:369 and
already driving the REAL eval_val with partition-invariance legs):
[0, 1, 9, 10] -> 5 not 1 through the real eval_val; odd control
[0, 1, 9] -> 1 exactly (byte-identity for odd N); batch/chunk
invariance of the median through real eval_val; an equal-fraction
epoch pair where lower-median and true-median rank opposite models
and the true-median model must win the tie-break; a plateau-
scheduler spy asserting the stepped value equals the reported and
persisted median; mutation arm retaining Tensor.median() must fail.

## UNIT 14 REOPENED AGAIN (45M-58, eleventh batch): increment (f) — validate the published reductions, not just the rows

CONFIRMED (Fable, 2026-07-12, live through the REAL function).
eval_val's landed chokepoint validates every per-sample chi2 (the
increment-(e) domain predicate raises on bad ROWS), then publishes

    mean   = c.mean().item()     # raw float32 reduction, unvalidated

(training.py:1539 at HEAD 420bce2). PyTorch's float32 mean forms a
float32 sum: eight finite rows near 1e38 have a true sum of 8e38 >
float32 max, so the intermediate overflows in ANY summation order and
the published mean is Inf AFTER the row guard declared the set valid.
Reproduced through the real eval_val (cocoa python, torch 2.6.0 CPU,
duck-typed loss returning 1e38 rows):

    rows: 8 x 1e38 float32, all finite; domain predicate: PASS
    published median = 9.999999680285692e+37   (order statistic, fine)
    published mean   = inf
    float64 reference mean = 9.999999680285692e+37

This violates unit 14's own clause 4 ("mean/median/fractions must be
finite before the publication" — the clause list above): increments
(a+b) implemented clause 1 (rows), nothing implemented clause 4 for
the reduction itself. Distinct from 45M-47 / increment (d), which
repaired the TRAINING-epoch loss accumulator; this is the validation
reporting path. The mean is appended to histories, plotted, and
persisted. eval_source_chi2 returns per-row (params, dchi2) with no
scalar reduction, so (f) is scoped to eval_val's published outputs.
Honest scoping: the median (order statistic) and the boolean
fractions (bounded by 1) cannot overflow — the mean is the vulnerable
reduction — but the post-reduction check covers all three because it
is one line each and because unit 60's even-N midpoint in float32
could itself overflow at extreme scale (the float64 helper covers
both).

Contract, increment (f):

1. eval_val computes its published reductions in float64 on the CPU
   tensor: the mean now; the median through unit 60's shared ordinary-
   median helper when it lands (same visit — do not build two
   helpers). The threshold fractions are means of {0,1} and stay
   exact; they are validated with the rest.
2. EVERY published scalar/vector (mean, median, frac) is validated
   finite AFTER reduction, before return; failure RAISES naming the
   reduction, the side ("validation"), and the offending value.
   Never a sentinel repair — an infinite mean is a refused
   evaluation, not a big number.
3. Ordinary-range results are numerically unchanged to the documented
   tolerance: the float64 mean of float32 rows differs from the old
   float32 mean at rounding level (~1e-7 relative). Histories are NOT
   byte-identical — USER-VISIBLE, declared.
4. The row-level guards ((a+b), (e)) are untouched; (f) is clause 4
   made real.

Gate Part I (finite-contract, workstation; CPU AND the CUDA lane):

- through the REAL eval_val: eight finite float32 rows near 1e38 ->
  the published mean is finite and equals the float64 reference
  within the documented tolerance; median and fractions finite;
- the same value reaches the training-loop history append (the
  return contract is the history value);
- mutation arm: restoring the float32 c.mean() must FAIL on a lane
  where the float32 sum overflows; a backend that happens not to
  overflow is recorded as a CONTROL — the contract is never
  backend-dependent;
- ordinary-scale positive control within the documented tolerance.

Placement: (f) rides the SAME eval_val visit as unit 60, in the
pipeline slot 50(+60+14f), after queue 43. Unit 14's closure claim
(a-e) stands for its increments; the unit stays open on (f) only.

#### Units 60 + 14(f) COMPLETE resume (2026-07-12, Opus) — committed 4846fdd, eval_val half of the 50-bundle

Units 60 + 14(f) self-committed on the branch as 4846fdd (batch grant,
pending audit) as ONE eval_val visit (the RULING's "same visit -- do not
build two helpers").

- Unit 60 (45M-57): `ordinary_median(values)` in training.py is the ONE
  shared 50th-percentile reduction (torch.quantile(., 0.5) in float64 --
  the mean of the two central values for even N, byte-identical to
  torch.median for odd N, with the ~2^24 element cap documented). eval_val's
  `median = c.median().item()` -> `ordinary_median(c)`; the scheduler feed,
  the tie-break, and the histories consume this returned median, so the one
  fix propagates to all four (no separate edits at those sites).
- Unit 14(f) (45M-58): eval_val computes the mean in float64
  (`c.to(torch.float64).mean()`) -- a float32 mean of rows near the float32
  max overflowed to Inf AFTER the row guard -- and
  `_validate_published_reductions` refuses any non-finite published mean /
  median / fraction before return (clause 4). Row-level guards (a/b/e/h)
  untouched.
- The five gate reference sites migrated in this unit: ge_c_eval_bs.py (the
  Part 1 reference + a NEW Part 1b with the unit-60 red legs), finite_contract.py
  (two references + the float64 mean), cmb_smoke / bsn_smoke / mps_smoke (the
  mean-predictor reference). `import math` added to training.py.

Gate (unit 60 red legs, in ge_c_eval_bs.py Part 1b, board-run): helper parity
(even ordinary=5 vs Tensor.median=1; odd=1), the REAL eval_val even-median 5 /
odd-median 1, batch invariance across bs 1..4, the Tensor.median mutation
caught. Part 1 (partition invariance) reference migrated so it still passes.

Verified on Cocoa torch (CPU): probe_median_reductions.py 11/11 (helper
parity, real eval_val even=5/odd=1, batch invariance, the float64 mean finite
at 1e38 scale where float32 overflows, the post-reduction guard refusing an
Inf mean / NaN median / non-finite fraction); ge_c_eval_bs Part 1 + Part 1b
PASS on CPU.

USER-VISIBLE, declared: even-n_val medians and means change (reported,
persisted, plotted); plateau timing / tie-breaks can shift; odd-n_val medians
exactly unchanged; ordinary-scale means differ at ~1e-7 rounding level.

Workstation owed (user-run): finite-contract + the family smoke gate reruns
(need cosmolike / real CAMB -- cannot import on Mac).

REMAINING in the 50-bundle: unit 50 (45M-38, epoch truth under VRAM chunking,
notes/training-stack.md:1131 -- CRITICAL, the training_loop_batched epoch
reduction, a DISTINCT surface from eval_val) is NOT in this commit. Queue
after 50: 52 (propose-first head-padding) -> 55 (45M-46) -> 22(+20) ->
13(+01, label 45M-01).

## UNIT 14 REOPENED (45M-60 + addendum, twelfth batch): increment (g) — the chi2-domain band scales with the contraction WIDTH, not the product count

Post-landing audit of 420bce2, CONFIRMED. Landed state:
CosmolikeChi2._chi2_n_terms returns w*w on the dense base
(losses/core.py:254 region, "w^2 products" in its docstring);
CmbDiagonalChi2 overrides to w (losses/cmb.py:215); _chi2_neg_band
multiplies 32 * eps(dtype) by that count (core.py:100 region);
_chi2_domain clamps every within-band negative to EXACT 0. The
production bands that fall out (computed live, float32):

    w =  780:  w^2 band = 2.32086     width band = 0.002975
    w = 1000:  w^2 band = 3.81470     width band = 0.003815
    w = 3000:  w^2 band = 34.33228    width band = 0.011444

Under w^2, a dense production loss returning chi2 = -2.0 is not
refused — it becomes the best possible score. That reintroduces the
false-crowning failure increment (e) exists to close. And the
shipped gate never exercises the production band: the negative and
band-edge legs use PoisonChi2, which omits _chi2_n_terms, so
eval_val substitutes n_terms = 1 and only the 1e-6 floor is tested
(finite_contract.py:913-914).

The record, adjudicated precisely: the eighth-batch ruling text
said "n_terms = the per-row count of summed products in the active
contraction (n_dv for the plain whitened form; the documented
equivalent for the rescaled/transfer forms)". The parenthetical
anchored WIDTH (n_dv); the head phrase, read literally against a
dense r^T Cinv r, yields w^2. The Implementer resolved the
ambiguity to w^2/w and documented the resolution openly (commit
message and resume note) — NOT a silent redefinition — but the
resolution contradicts the n_dv anchor, and the consequence test
decides. The ambiguous phrase was the Architect's; the contract is
hereby revised WITH the derivation the red team demanded.

RULING: n_terms := the per-row kept WIDTH w for EVERY
CosmolikeChi2 family. Derivation (depth, not count): the band only
governs values near zero; there the computed chi2's roundoff is
bounded by (accumulation depth) * eps * (sum of term magnitudes).
The dense contraction executes as a matvec — w INDEPENDENT
length-w sums — followed by one length-w dot, so the final
accumulated chain is ~w deep (torch's pairwise/blocked reductions
make even w conservative), and near a small chi2 the term
magnitudes are themselves small; the flat-chain w^2 model both
overcounts the depth and ignores the small-term structure.
Empirical anchor: the valid roundoff negatives that motivated
45M-53 sat at ~1e-6 — three-plus orders inside the width band at
every production w — while -2.0 / -4.0 are refused at every
production width. _CHI2_NEG_KAPPA = 32 and the 1e-6 floor stand.
GROWTH CLAUSE: the band may only ever be WIDENED by measured valid
controls (the SPD leg below) plus a recorded forward-error
derivation — never by convenience.

ADDENDUM adopted (the metric census): with width uniform, ONE
definition lives on the base class — n_terms = the geometry's kept
per-row width — and the CmbDiagonalChi2 override is retired as
redundant. ScalarChi2 (explicitly a diagonal sum of n_out squared
standardized residuals, losses/scalar.py:28) becomes correct
automatically; the addendum caught it silently inheriting the
dense w^2 rule today. Every loss family documents the metric its
chi2 actually executes at the class; test doubles used in
production-band gate legs DECLARE their width explicitly — the
silent hasattr fallback to n_terms = 1 (training.py:1528/:1618)
remains only for bare doubles outside production-contract legs.

Gate Part H amendments (board-listed finite_contract.py,
workstation; no separate unlisted script):

- a production-surface leg using an actual CosmolikeChi2-class
  loss whose dest_idx has realistic dense width (>= 780);
- -2.0 and -4.0 RAISE at realistic widths, before ranking or
  normalization;
- both sides of the ACTUAL production band, with the exact band
  value reported in the leg output;
- mutation arm: restore w*w — the realistic-width negative leg
  must fail;
- scalar-width leg: ScalarChi2 yields n_terms = n_out;
- mechanical subclass census: every CosmolikeChi2 subclass returns
  its geometry's kept width (a future diagonal family cannot
  silently inherit a wrong rule);
- an ill-conditioned SPD VALID control measuring where genuine
  roundoff negatives land — they must fall inside the width band
  (unit 11's future SPD guards are complementary, never a
  substitute).

PRIORITY: increment (g) PREEMPTS queue 43 — a live false-crowning
hole on the branch beats a design unit; the revision is surgical.
Unit 14 stays open on (f) + (g); pipeline slot for (g) is
immediately next.

## UNIT 61 (45M-59, twelfth batch): the learning-curve figure must represent a perfect zero

CONFIRMED (Fable, 2026-07-12). plot_learning_curves documents
f(delta-chi2 > threshold) as the plotted result, then
unconditionally selects ax.set_yscale("log")
(emulator/plotting.py:304). A valid fraction may equal EXACTLY 0 —
no validation row exceeded the threshold, the best possible
outcome — and zero has no logarithm: the most successful point is
dropped or clipped, and the saved figure no longer represents the
data supplied to it. The same module already implements the
correct policy in plot_sweep_curve: "y is logarithmic when every
fraction is positive, linear otherwise (a perfect 0.0 point would
break a log axis)" (docstring, ~:324) with the conditional
np.all(fr[np.isfinite(fr)] > 0) -> log at :372-373. Two public
plotting paths disagree on the same quantity; reachable through
the public sweep/learning-curve outputs.

Contract (the red team's six clauses, one addition):

1. Validate every training size finite and strictly positive, and
   every fraction finite in [0, 1] — RAISE before figure
   construction; matplotlib warnings are not schema validation.
2. Sort accepted curves by training size, as today.
3. Log y only when ALL plotted fractions are strictly positive; if
   any accepted fraction is zero, use the zero-capable scale (the
   existing plot_sweep_curve linear policy is the minimal
   consistent rule).
4. A zero marker stays visible at its exact coordinate, including
   as the final, scientifically decisive point.
5. target validated finite in [0, 1]; a zero target remains
   representable.
6. The docstring names the conditional scale (no unconditional
   log-log promise).
ADDITION (Architect): ONE shared scale-decision helper called by
BOTH plot_learning_curves and plot_sweep_curve, so the two public
paths cannot drift apart again — the parity leg then proves the
sharing, not a coincidence.

Gate legs (CPU-only, matplotlib Agg; the Implementer proposes the
home — a small board-listed check or the existing plot coverage —
at build): positive-only curve retains log y; {100: .5, 1000: .1,
10000: 0} uses the zero-capable scale with the marker present at
y = 0; an interior zero is not silently bridged away; fractions
below zero / above one / NaN / Inf raise before figure
construction; nonpositive or nonfinite training sizes raise;
mutation arm restoring the unconditional set_yscale("log") must
fail the perfect-zero leg; parity leg proving both public paths
make the same scale decision on the same finite fractions.

Placement: campaign phase (CPU-only, independent); no preemption.

## UNIT 14 increment (g) AMENDED (45M-60 second addendum, thirteenth batch): the band derives from the COMPUTE dtype, never a storage upcast

CONFIRMED (Fable, 2026-07-12). eval_source_chi2 computes each chi2
chunk in the model/loss compute dtype (normally float32), then

    dchi2_t = torch.cat(chunks).double()          (training.py:1608)
    band = _chi2_neg_band(dchi2_t.dtype, n_terms) (training.py:1620)

The upcast cannot undo float32 contraction roundoff; it only
relabels the dtype the validator reads. Measured: at w = 780 the
float64-eps band is 5.5e-12 raw -> the 1e-6 floor, while the
float32 band (post-(g) width rule) is 0.002975; at w = 3000,
2.1e-11 -> floor vs 0.011444. So training's _reduce and eval_val
normalize a -5e-4 roundoff negative to exact 0 while the public
diagnostic/sweep scorer REFUSES the same value — one score, two
verdicts — and the comment at :1609-1615 explicitly claims the
same predicate and band. An executed contradiction, not a policy
choice.

Amendment (folds into increment (g); rides the board-listed
finite-contract gate, no new gate file):

1. Capture the chi2 COMPUTE dtype before any storage/reporting
   cast; derive the band from that dtype.
2. Validate/normalize in the compute dtype; convert the ACCEPTED
   result to float64 NumPy for reporting.
3. Numerical provenance is never inferred from a tensor that has
   merely been upcast.
4. The same ordering applies to increment (h)'s shared diagnostic
   helper (the floor arm at diagnostics.py:226 has the identical
   .double()-before-interpretation shape).
5. Gate leg: a real float32 loss family at a value between the
   1e-6 floor and its adjudicated float32 band — _reduce, eval_val,
   and eval_source_chi2 must give ONE verdict and one exact-zero
   normalization.
6. Mutation arm: restoring .double() before _chi2_neg_band must
   fail.
7. A genuine float64-compute control: a loss actually evaluated in
   float64 still receives the float64 band.

#### Increment (g) COMPLETE resume (2026-07-12, Opus) — committed cf1ab16, unit 14 open on (h)

Increment (g) self-committed on the branch as cf1ab16 (batch grant, pending
Architect audit): the width rule (twelfth batch) AND the compute-dtype band
provenance (thirteenth batch, second addendum). This resume is a follow-up
commit because the runner's concurrent notes commits repeatedly clobbered the
uncommitted resume. Sequencing note: my relayed handoff ordered (g) before 43;
I built (g) first and disentangled it from the queue-43 loss-side WIP (the
cmb.py (g) diff is ONLY the override removal), which resolves the entanglement
the thirteenth batch cited for "43 first". (h) is next, then queue 43.

Width rule (45M-60):
- losses/core.py: `CosmolikeChi2._chi2_n_terms` returns the kept width
  `int(self.dest_idx.numel())` (was w*w); the docstring carries the
  depth-not-count derivation + the GROWTH CLAUSE. `_chi2_neg_band` + the module
  comment now read "reduction depth = kept width w".
- losses/cmb.py: the redundant `CmbDiagonalChi2._chi2_n_terms` override removed
  (a class comment documents the inherited width band; no CMB behaviour change).
- losses/scalar.py: ScalarChi2's docstring records it now inherits the width
  band (the addendum's catch: it silently carried the dense w^2 rule before).

Second addendum (45M-60, thirteenth batch):
- training.py `eval_source_chi2`: the band derives from the COMPUTE dtype
  (`c_compute = torch.cat(chunks)`, no `.double()` before the band); validate /
  normalize in the compute dtype, cast the ACCEPTED result to float64 for
  reporting only. Fixes the one-score-two-verdicts split (the float64 upcast
  floored the band to 1e-6, refusing a roundoff negative _reduce / eval_val
  normalize to 0).

Gate (gates/checks/finite_contract.py, board-listed, no new file):
- `check_chi2_band_production` (45M-60): a REAL CosmolikeChi2 subclass at width
  780 -- -2 / -4 RAISE (band 0.002975 reported); both band sides (half
  accepted, double raises, above the 1e-6 floor); a w^2-restoring mutation arm
  (band 2.32086) that must NOT raise; a scalar-width leg (n_out); a mechanical
  subclass census; an ill-conditioned SPD roundoff control. All via eval_val
  (float32 band).
- `check_chi2_band_dtype_provenance` (second addendum): _reduce / eval_val /
  eval_source give ONE verdict on a value between the 1e-6 floor and the float32
  band; the restored .double() upcast splits them (mutation); a float64-compute
  loss gets the tight float64 band.
- board.py maps + the gate docstring name 45M-60 + the second addendum.

Mac verification (raw): py_compile OK on core.py, cmb.py, scalar.py,
training.py, finite_contract.py, board.py; probe_width_band.py 9/9 and
probe_band_dtype.py 4/4 on the REAL losses + eval_val + eval_source_chi2 +
_reduce (Cocoa torch 2.6, CPU) -- n_terms 780 not 608400, bands 0.002975 /
2.32086, eval_val refuses -2 / -4, the w^2 mutation swallows -2, the three
boundaries agree on -5e-4 (float32 compute) and the float64-compute loss gets
the float64 band. The torch legs ride the workstation finite-contract gate
(still 33/33, zero skips = compile mandatory).

Unit 14 stays OPEN on (h) 45M-61 (the diagnostic score boundary: the shared
public score-domain helper + the four producer sites + the diagnostics gate)
and (f) 45M-58 (float64 published reductions; rides 50). Next: (h), then queue
43 under the QUEUE 43 RULINGS, then 50(+60+14f) -> 52 -> 55 -> 22 -> 13.

## UNIT 14 REOPENED (45M-61, thirteenth batch): increment (h) — the diagnostic score boundary

CONFIRMED (Fable, 2026-07-12). The finite-chi2 contract stops
before the "data-only floor": local_linear_floor computes its
interpolation-floor score by calling chi2fn.chi2 DIRECTLY
(diagnostics.py:226-227, including the same
.double()-before-interpretation upcast) and immediately interprets
the unchecked values — f_floor / f_hard via dchi2_floor > 0.2
(:238, :240) and median_floor via np.median (:241) — while only
the MODEL arm (:230-233) passes through the newly guarded
eval_source_chi2. _chi2_domain and _chi2_neg_band appear NOWHERE
in diagnostics.py (untruncated census, 2026-07-12). Three more
direct producer sites: :414, :621, :745 (the CMB / grid / grid2d
public residual functions). Increment (e) therefore established
two different definitions of a valid diagnostic chi2 inside one
returned record.

Reachability and the concrete wrong result (adjudicated on the
validator boundary + arithmetic; the red legs drive the real
path): DataVectorGeometry.from_state splats state straight into
the constructor (output.py:249 `cls(device, **state)`), which
stores Cinv and slices Cinv_sq with NO positive-definiteness check
(:163-186); geometry-totality unit 11 is queued, not landed. A
one-coordinate state with Cinv = [[-1]] and a unit floor residual
gives dchi2_floor = -1: since (-1 > 0.2) is False, f_floor = 0.0 —
the impossible negative "data-only floor" is reported PERFECT —
and median_floor = -1; NaN rides the same NaN-comparison-False
path. Even after unit 11 lands, float32 contraction roundoff is
the reason (e) built the shared band; the floor must use it too.
Distinct from the thirteenth-wave statistics-manufacture-NaN
extension (already folded at ledger entry 9): there the statistics
corrupt finite inputs; here a chi2 PRODUCER returns an invalid
score and the diagnostic boundary fails to apply the
already-landed score-domain contract.

Contract (the red team's clauses adopted whole; no new tolerance):

1. ONE shared public score-domain helper owned beside _chi2_domain
   in losses/core.py; it accepts the LOSS OBJECT and derives the
   band from that family's adjudicated term count (increment (g))
   and the compute dtype captured before any storage cast (the
   (g) second addendum ordering).
2. dchi2_floor runs through it BEFORE any threshold, median,
   dense-decile, plotting, or persistence operation.
3. The three other direct sites (:414, :621, :745) are censused:
   every public family diagnostic function validates its OWN chi2
   vectors — none may rely on a driver having happened to run
   coverage_diagnostic first.
4. A materially negative or non-finite score RAISES naming: the
   diagnostic name, the score producer (local-linear floor, cmb
   residual, ...), the bad-row count and positions, the minimum,
   and the band. A within-band negative becomes exact zero under
   the same rule as training/evaluation.
5. Unit 11's geometry validation is defense in depth, not a
   substitute for score-boundary validation.
6. Unit 9's honest-unavailability status governs statistics that
   are mathematically unavailable; it must NOT convert a corrupted
   chi2 into "unavailable" and continue.
7. Valid positive diagnostic output remains byte-identical.

Red legs (torch; a board-listed diagnostics gate under
gates/checks/ — the Implementer proposes the home, new small gate
or the existing family-diagnostics coverage; GPU workstation lane
plus a CPU lane wherever supported):

- the REAL local_linear_floor driven with the reachable
  one-coordinate accepted geometry whose floor score is -1:
  current code returns f_floor = 0.0, repaired code must REFUSE
  before computing it;
- a materially negative model-independent floor beside an
  otherwise finite/valid model arm — proves the FLOOR guard, not
  eval_source_chi2, catches it;
- NaN and +/-Inf floor scores refuse;
- values immediately on both sides of the corrected
  production-family band: exact-zero normalization vs refusal;
- mutation arm: deleting ONLY the floor guard recreates the false
  f_floor = 0;
- each of the CMB / grid / grid2d public residual functions gets
  one corrupt-score refusal and one valid-output identity control;
- the loss-family term-count census from the (g) addendum rides
  here too: the diagnostic must never silently use the
  fallback-one band.

Placement: increment (h) rides WITH (g) as one visit, AFTER queue
43 lands (the preemption was relaxed: 43's loss side is built and
verified uncommitted in losses/cmb.py, the same file (g) edits;
finishing 43 avoids a half-unit and a same-file collision). Unit
14 stays open on (f) + (g) + (h).
## 45M-89 red-team documentation amendment: diagnostics must separate an estimator from a scientific verdict (2026-07-12)

The current `diagnostics.py` prose teaches several heuristic outputs as if
they establish causation.  `coverage_diagnostic` says a positive rank
correlation means the floor is data coverage rather than the model, while its
Boolean verdict is the unmotivated conjunction `median_bad > median_good` and
`rho > 0.1`.  The same file hard-codes the good/bad boundary 0.2, dense/sparse
deciles, a `1e-4` log floor, `k_nn=8`, a local-linear `k_nn=40`, and a CMB
period fallback 50.  `plotting.py` adds percentile clipping and histogram-bin
floors/ceilings.  Comments call some of these “readable” or “pure hardness”
without deriving the choices or stating their sensitivity.

Required documentation contract, folded into the queued diagnostics-totality
unit rather than a second diagnostic implementation:

- Define k-nearest-neighbor distance, Spearman rank correlation, local linear
  regression, percentile, decile, and $R^2$ in plain language at the first
  executable use.  State shapes and units.
- A local-linear estimate is one comparator under locality and smoothness
  assumptions.  It is not a mathematical lower bound and not “the best” a
  smooth method can extract.  Rename prose and plot labels accordingly.
- Descriptive statistics remain descriptive.  A correlation and a median
  split can support “consistent with coverage limitation”; they cannot prove
  that the model is not responsible.  State competing explanations and the
  experiment that would distinguish them.
- Every numerical control becomes a named argument or named module constant
  with a derivation, units, and one sensitivity check.  The physical
  $\Delta\chi^2=0.2$ acceptance threshold may remain shared from the run; the
  0.1 correlation threshold, neighbor counts, deciles, floors, period, and
  plot clipping cannot remain anonymous literals.
- Teach device movement exactly: which arrays are copied from NumPy to
  float32 device tensors, which calculations return to CPU and float64, and
  why SciPy operates on host arrays.
- The diagnostics-totality status/reason record owns empty groups, constant
  features, zero truth crossings, and unavailable regressions.  Documentation
  never describes NaN as an ordinary scientific number or lets plotting
  silently reinterpret it.
- A gate varies each newly named heuristic on a finite known-answer dataset
  and proves that a categorical verdict cannot flip without the report naming
  the changed control.  Existing numerical-totality legs remain mandatory.

This amendment is both scientific and didactic: a novice must learn what the
diagnostic measures, while an expert must be able to see which part is a
measurement and which part is an interpretation.

#### Increment (h) COMPLETE resume (2026-07-12, Opus) — committed 3f47d86, diagnostics-domain gate green

Increment (h) self-committed on the branch as 3f47d86 (batch grant, pending
Architect audit). The diagnostic score boundary now goes through ONE shared
public helper.

Contract delivered whole (the seven clauses):
- `screen_chi2(chi2, loss, label, positions=None)` in losses/core.py beside
  `_chi2_domain`: derives the band from `loss._chi2_n_terms()` (increment (g),
  getattr default 1) and the chi2 COMPUTE dtype (the (g) second addendum --
  the tensor is passed pre-.double()), applies `_chi2_domain`, RAISES naming
  the boundary + rows + minimum + band, or returns the within-band-normalized
  c_norm (exact 0). It raises rather than converting to "unavailable" (unit 9)
  and is applied regardless of any upstream geometry check (unit 11 defense in
  depth). Valid positive output byte-identical.
- Consolidation: eval_val + eval_source_chi2 (training.py) now CALL screen_chi2;
  the private `_report_chi2_domain` is retired (its message is preserved
  byte-for-byte inside screen_chi2, so finite_contract's message assertions
  still hold).
- diagnostics.py: local_linear_floor (:262) screens the floor in its compute
  dtype BEFORE f_floor / median_floor (its `.double()` ordering fixed); the
  three residual producers (cmb/grid/grid2d) accumulate the compute-dtype chi2
  per chunk and screen after concat through `_screen_diag_chi2` (a thin DRY
  wrapper that calls screen_chi2), never a per-chunk `.double()`.

Gate (NEW, board-listed): gates/checks/diagnostics_domain.py (DIAG-A, torch
CPU, no cosmolike/CAMB) + gate_diagnostics_domain + the Gate() entry in
board.py. 20/20 GREEN on Cocoa torch:
- screen_chi2 unit: valid byte-identical, within-band roundoff -> exact 0,
  materially negative / NaN / +Inf / -Inf refused naming row + band, the
  fallback-1 band floor still rejects, and the term count widens the band
  (no silent fallback-1);
- the REAL local_linear_floor: valid control returns finite f_floor; a
  reachable negative floor (a _FloorOnlyNegChi2 corrupting call #1 = the floor,
  the model arm valid) REFUSED before f_floor naming "local-linear floor"; a
  NaN floor refused; the mutation arm (diagnostics.screen_chi2 monkeypatched to
  a passthrough) recreates the false f_floor = 0, median_floor = -1e3;
- the REAL cmb_residual_diagnostic: valid control + a corrupt-score refusal
  naming "cmb residual";
- a source census (AST): cmb/grid/grid2d residual all route through
  _screen_diag_chi2, local_linear_floor calls screen_chi2, _screen_diag_chi2
  delegates to screen_chi2, and no residual producer keeps the raw
  .double().cpu().numpy() chi2 path.

Scope note (honestly flagged): the grid / grid2d residual LIVE corrupt-score
refusals are covered by the CENSUS leg (identical _screen_diag_chi2 path,
proven live for CMB) rather than a separate live fixture -- the shared boundary
is exercised live once and the other two are proven to route through it. If the
Architect wants grid/grid2d live refusals too, they are a small add (build a
GridGeometry + Grid2DGeometry + ScalarChi2 fixture).

Workstation owed (user-run): the finite-contract gate rerun (Part A/C/H
eval_val / eval_source_chi2 now route through screen_chi2, message unchanged --
cannot import on Mac: geometries.output -> cosmolike) and the family smoke
gates (residual internal accumulation changed; valid output identical).

Unit 14 stays OPEN on (f) 45M-58 (float64 published reductions; rides unit 50).
Next in the queue: 50(+60+14f) -> 52 -> 55 -> 22(+20) -> 13(+01).

## Structured evidence map — gate contract anchors (45M-72 foundation)

The board's structured evidence map (`Gate.evidence`) pins each migrated
gate to a stable, runner-validated anchor in its home note; the mechanism
and the audited rollout are documented in `gates-and-board.md`. The
diagnostics gate anchors here:

<a id="diag-a-diagnostics-domain"></a>
**diagnostics-domain (DIAG-A) — the diagnostic score-domain boundary.** The
shared `screen_chi2` helper (valid input byte-identical, within-band
roundoff pulled to exact 0, a materially negative / NaN / +-Inf score
refused naming the boundary + rows + band, the fallback-1 floor, the
width-scaled band); the real `local_linear_floor` (a reachable negative
floor refused before `f_floor`, a NaN floor refused, a valid control, and
the guard-bypassed mutation that recreates the false `f_floor = 0`); the
real `cmb_residual_diagnostic` (corrupt-score refusal + valid control); and
the grid / grid2d producer census through the one shared boundary.

## UNIT 79 (20M-12, 2026-07-13): roughness is a CMB-family capability — eligibility by explicit family, never by inheritance

Finding (red team, CONFIRMED; witness reproduced through the shipped
implementation): PCEResidualDiagChi2 inherits CmbDiagonalChi2
(losses/pce.py:333) including configure_roughness, and
training.py:2733 decides family by hasattr — its own comment says
the method's presence identifies "a CMB loss" — so scalar / grid /
grid2d NPCE runs legally carrying train_args.loss.roughness smooth
redshift bins, flattened (z,k) cells, or unrelated named scalars as
if they were consecutive multipoles (alternating length-7 witness:
objective 7 -> 13.4512 at period_cut=5, lam=1; a two-output scalar
run instead crashes at first batch on reflect padding).

Contract (ratified): (1) roughness eligibility is an explicit
scientific-family capability, not method presence or incidental
inheritance; (2) only data.cmb may carry train_args.loss.roughness;
(3) scalar, grid, grid2d, and ordinary cosmolike runs reject the
block at configuration validation — before staging, PCE fitting,
model construction, or training; (4) PCEResidualDiagChi2 may keep
the shared diagonal metric, but the CMB-only training feature does
not leak through the shared base; (5) the valid CMB+NPCE+law-none
path retains the existing roughness computation exactly; (6) error
prose explains the filter assumes an ordered, uniformly interpreted
multipole axis — output-vector adjacency alone does not create that
meaning.

Legs (ratified; CPU Torch, board-listed): scalar/grid/grid2d NPCE +
roughness each refuse at validation; plain non-NPCE controls refuse
identically; CMB+NPCE+law-none accepted and matches a direct
known-answer penalty; the alternating seven-coordinate example
remains 7 (rejected before loss construction, never 13.4512);
two-output scalar refuses at validation instead of crashing; a
mutation restoring the hasattr decision must accept a non-CMB NPCE
configuration and fail the gate. Sequencing: small standalone
training-truth refusal; may ride any nearby training.py landing.

## UNIT 80 (20M-13, 2026-07-13): one physical-contraction owner — the residual is cast to the precision tensor's dtype at the boundary, nowhere else

Finding (red team, CONFIRMED): float64 output geometries are
documented and recommended (output.py:76-79) and store Cinv in the
geometry dtype, but the two physical-composition losses force the
physical truth to float32 (pce.py:266; transfer.py:291, :294) and
contract directly with the stored-dtype precision (pce.py:310-311)
— every float64 PCE-ratio and physical-transfer configuration
crashes (kept and full, sum and gain) before any chi2 exists; the
whitened route is dtype-aware and fine.

Contract (ratified): (1) network outputs and staged packed targets
STAY float32; (2) immediately before every physical Mahalanobis
contraction the physical residual is cast to the EXACT dtype and
device of the precision tensor it contracts with — kept and full
identically; (3) the returned chi2 dtype follows Cinv_sq / Cinv (a
float64 geometry yields a float64 chi2, which the unit-14 screen's
compute-dtype provenance already accommodates); (4) no wholesale
model/batch float64 conversion — this is a narrow loss-boundary
cast; (5) the float32 path is byte-identical; (6) ONE shared
physical-contraction helper owns the cast + einsum, used by PCE
ratio, transfer sum/gain, plain and factored transfer, and any
future physical-space loss — no per-loss drift.

Legs (ratified; CPU Torch, board-listed): float64 PCE-ratio kept +
full contractions against direct known answers; float64 physical
transfer sum + gain on a plain base; the same on a factored base;
the whitened-space float64 control; float32 controls proving
unchanged values AND dtypes; returned-dtype assertions for both
precision choices; a mutation arm restoring the direct mixed-dtype
einsum must reproduce the runtime failure. Sequencing: lands in the
transfer campaign WITH unit 77 as one algebra increment — unit 80's
contraction owner is where unit 77's composition owner contracts.

## UNIT 80 AMENDED (20M-13 addendum, 2026-07-13): geometry precision has one end-to-end owner — head basis buffers cast at their boundary

Finding (red team, CONFIRMED): the structured heads register
W_fd / W_df directly from the geometry's float64 eigenvector/scale
tensors (plain.py:577-594; mirrored in ia.py), so a float64
DataVectorGeometry crashes ResCNN / ResTRF / TemplateResCNN /
TemplateResTRF at y @ W_fd ("expected m1 and m2 to have the same
dtype") before any loss executes — the public Python geometry API
reaches it even though the YAML surface does not yet expose the
dtype knob.

Amendment (binding, lands IN the unit-80 increment): (1) model
computation stays in the declared model compute dtype (normally
float32); (2) structured-head basis-transform buffers are derived or
cast explicitly into the trunk-output dtype at their owned
composition boundary; (3) the loss geometry keeps its requested
precision — float64 Cinv stays float64 for the unit-80 contraction;
(4) the forward and inverse basis transforms get an independent
known-answer check; (5) default float32 geometry/model behavior
bitwise identical; (6) all four structured families complete forward
AND backward under a float64 output geometry; (7) a mutation
retaining float64 W_fd/W_df beside a float32 trunk must red.
"supported geometry precision" has ONE owner from model head through
physical contraction.

## UNIT 88 (20M-24, 2026-07-13): capacity tokens are acquired before any job-sized allocation — the budget gate contains what it budgets

Finding (red team, CONFIRMED; live instrumentation through the real
run_gpu_pool): lanes run setup_fn — which stages full train +
validation data and builds the geometry (:131-138) — before any
token is acquired (scheduling.py:242 vs :252-259), so four charged
resident experiments coexist under a token model that promises
exclusivity; the estimator explicitly charges the resident data term
that setup allocates.

Contract (ratified): (1) NO job-sized GPU allocation before that job
owns its capacity tokens; (2) setup splits into genuinely small
permanent lane setup + token-scoped per-job staging, OR tokens are
held for the complete lifetime of one lane-local staged experiment —
never four resident experiments preserved to preserve today's
function split; (3) genuinely permanent per-lane CUDA
context/model/geometry state is accounted separately and multiplied
by the ACTUAL lane count before remaining token capacity is
advertised; (4) the arithmetic is exact: a four-token job permits one
charged resident allocation on the GPU, two-token at most two,
one-token at most four — setup and execution COMBINED; (5) setup
failure, acquisition failure, and job failure release exactly what
they acquired, retaining the sibling-reaping/liveness contract; (6)
the banner reports permanent-per-lane bytes, token-scoped bytes,
lane count, and measured/derived concurrency — no "exclusive" or
packing claims the quantities do not support; (7) the N-train path's
validation staging receives the same ownership audit (the
hyperparameter path is the decisive current counterexample because
both stagings live in setup_fn).

Legs (ratified): a board-listed CPU allocation-counter fake through
the REAL run_gpu_pool proving maximum charged concurrency 1/2/4 for
4/2/1-token plans, with a mutation moving acquisition back below
setup (must red); a workstation CUDA leg with a deliberately tight
resident fixture whose four pre-fix setups cannot coexist but whose
token-scoped repaired execution completes, reporting raw peak
allocated/reserved bytes. Placement: the scheduler/pool campaign
(beside the bakeoff-liveness item and unit 55); blocks --gpu-pack
production sweeps.

## UNIT 89 (20M-25, 2026-07-13, HIGH): every training invocation establishes the COMPLETE loss-object state — absence clears, never inherits

Finding (red team, CONFIRMED; executed on the real repeated-run
path): configure_roughness fires only for a non-null resolved block
(training.py:2732-2741) and nothing clears _rough/_rough_lam
(losses/cmb.py:205-223), while the sweep driver reuses one staged
experiment + chi2fn per GPU lane: after an enabled point, a later
disabled point still optimizes lam = 1.0 roughness (13.4511995 vs
fresh 7.0 on the alternating seven-multipole residual). Sweep order
and lane assignment become scientific variables.

Contract (ratified, with the census widening): (1) every training
invocation establishes the COMPLETE loss-object state — roughness
OFF is established explicitly, never inherited; (2) ONE owner
replaces or clears the roughness state; block absence means CLEAR,
never leave-unchanged; (3) enabled->disabled, disabled->enabled,
repeated values, and lane/order permutations match isolated
fresh-experiment controls; (4) a failed sweep point leaves no state
affecting the next; (5) the resolved record describes the state
ACTUALLY installed on the loss object; (6) single-run enabled and
disabled numerics byte-identical; (7) CENSUS: every conditional
configure_* on a loss object (configure_law, configure_rescaling,
transfer's configure_roughness) is audited under the same
establish-or-clear discipline in this unit's landing — one proven
leak is not treated as the only one.

Legs (ratified; board-listed Torch, CPU sufficient — the real
repeated-run/same-experiment path): enabled->disabled;
disabled->enabled; fresh-object equivalence; failure-then-valid
isolation; lane permutation; a mutation deleting the OFF reset must
red. Placement: with unit 55 in the repeated-training isolation
class (their gates share the same home); blocks production
hyperparameter sweeps over loss blocks; distinct from unit 79
(eligibility) and unit 55 (transfer-source lifecycle).
