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
