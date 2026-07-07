---
name: weight-ema-snapshot-coupled
description: "Design spec (Architect, 2026-07-06): weight EMA (Polyak averaging) coupled to the loop's snapshot/rewind state. Motivation: the sqrt-bs lr rule fixes optimization but not the implicit regularization small batches give for free (user measured frac>0.2 = 0.065 at bs 64 vs 0.10 at 768); EMA is the cheapest lever that stabilizes the tail metric at every bs. Core invariant (user-approved): ANYTHING THE REWIND UN-LIVES MUST ALSO DISAPPEAR FROM THE AVERAGE — the EMA joins the best-snapshot as one unit {theta, optimizer, theta_bar}, saved on every new best and restored on every lr cut, else an excursion of `patience` epochs owns ~99.7% of the average's mass at beta=0.999. Selection/metrics on theta_bar; plateau scheduler stays on the raw median (dynamics unchanged). Horizon in EPOCHS (beta derived from steps/epoch -> bs-invariant); EMA eval by in-place weight swap (foreach copy_, never .data reassign — CUDA graphs hold pointers); off by default = byte-identical loop."
metadata:
  node_type: memory
  type: project
---

# Weight EMA, coupled to the snapshot/rewind state

> Follow-on: [[ema-anneal-schedule]] adds an optional `ema.anneal` sub-block
> that ramps the horizon from 0 (beta(e) via the same derive_ema_beta, eager
> per-epoch float), defers the average past the terrible early era, and
> brings ema into the phase-block whitelist (per-phase / `ema: null` opt-out).

User request 2026-07-06, from the batch-size generalization gap
(frac>0.2: 0.065 at bs 64, 0.071 at 128, 0.10 at 768 despite the
sqrt-bs lr rule): EMA is the one lever that helps the tail metric at
every batch size for ~2-3% overhead. Design discussed and the coupling
invariant explicitly approved by the user ("I like this").

## The invariant (the whole design in one sentence)

Anything the rewind un-lives must also disappear from the average.
The loop's best snapshot becomes one unit {theta, optimizer state,
theta_bar}: saved together on every new best, restored together on
every plateau lr cut. Numbers: theta_bar = beta*theta_bar +
(1-beta)*theta per step; an excursion of patience ~ 15 epochs x ~390
steps at beta = 0.999 carries 1 - 0.999^5850 ~ 99.7% of the average's
mass — restoring only theta leaves theta_bar made of a rejected
trajectory.

## Design

1. **Config:** optional `train_args.ema` block, off when absent
   (absent -> the loop must be BYTE-IDENTICAL to today):

       ema:
         horizon_epochs: 3   # averaging window, in epochs

   `horizon_epochs` > 0 (int or float); unknown keys in the block
   raise (whitelist, the usual typo guard). "ema" joins
   SWEEPABLE_TOP_KEYS (a legitimate quality axis, unlike the derived
   eval batch).
2. **beta derived per-step from the horizon in EPOCHS** (bs-invariant
   by construction, same reasoning as derive_eval_bs):
   beta = 1 - 1/(horizon_epochs * steps_per_epoch), where
   steps_per_epoch = the loop's real full-batch count. Pure helper
   (exec-extractable), clamped to beta >= 0 for tiny runs.
3. **State:** theta_bar = one flat list of parameter clones
   (parameters only — no BN running stats in these models; index
   buffers are constant and stay the raw model's). Init theta_bar <-
   theta at the END of warmup (the high-lr thrash is not worth
   averaging); before that the loop runs exactly as today.
4. **Update:** after each optimizer.step(),
   `torch._foreach_lerp_(theta_bar, theta, 1 - beta)` under no_grad —
   a handful of fused launches (~tens of us vs the 2.05 ms step).
5. **Eval by in-place weight swap** (no second compiled twin): per
   epoch, after the train pass — eval raw (drives the plateau
   scheduler, dynamics unchanged today) -> tmp <- theta, theta <-
   theta_bar (foreach copy_), eval EMA (drives best-tracking and the
   printed val/med/frac) -> theta <- tmp. CRITICAL: in-place
   `copy_` only, never reassign .data or param objects — the
   compiled/CUDA-graph closures hold the parameter STORAGE pointers;
   swapping data in place keeps replay valid, repointing breaks it
   silently. One extra param-sized tmp buffer. With the derived eval
   batch, the second eval costs ~5 batches.
6. **Best-tracking and return:** best-tracking runs on the EMA
   frac[0] once EMA is live (before init, on raw, exactly as today —
   at init theta_bar == theta so the switch is continuous). The
   snapshot dict gains theta_bar; rewind restores all three; at the
   end the model is loaded with the BEST EMA weights (the shipped
   artifact).
7. **Two-phase runs:** at the trunk->head handoff the loop already
   restores phase-1 best and rebuilds the optimizer; theta_bar
   re-initializes to that restored theta (averaging across the phase
   boundary mixes two regimes). Phase 2's warmup delay applies again.
8. **Reporting:** one banner line when EMA is on, e.g.
   `ema: horizon 3 epochs (selection + metrics on the average;
   scheduler on the raw median)`; the epoch line's val/med/frac are
   the EMA's from init onward (they describe the model that would be
   shipped).

## Validation gate

- GM-A (pure, Mac, exec-extracted): the beta helper — horizon 3 x 390
  steps -> beta = 1 - 1/1170; tiny-run clamp; horizon <= 0 and unknown
  ema keys raise; block absent -> disabled sentinel. Config validation
  cases. Paste raw outputs.
- GM-B (static, Mac): AST/diff — every new branch is behind the
  ema-enabled flag (absent block touches NOTHING: the diff to the
  epoch loop is guard-gated); update sits after optimizer.step; init
  at warmup end; phase-boundary re-init; snapshot save/restore carries
  theta_bar with theta and optimizer; scheduler median comes from the
  RAW eval; best frac from the EMA eval; foreach copy_ swaps (grep: no
  `.data =` anywhere new); SWEEPABLE_TOP_KEYS gains "ema". Scans +
  whole-tree py_compile.
- GM-C (workstation, off-mode golden run): same YAML, same seed,
  ema absent, pre-change vs post-change build -> epoch lines
  bit-identical (the byte-identity gate; a launch-bound run is
  deterministic given fixed seeds and shapes).
- GM-D (workstation, on-mode smoke): a short bs=64 run with
  `ema: {horizon_epochs: 3}` — banner line present; EMA metrics track
  the raw metrics within noise early and are smoother late; a rewind
  event fires and the post-rewind EMA metrics jump WITH the raw ones
  (the invariant, visible in the log); returned model reproduces the
  best printed EMA frac on re-eval.

## Sequencing

training.py is also touched by the uncommitted eval-bs feature
([[eval-bs-decoupling]], D-E1/D-E2 pending): land AFTER the user
commits that unit, or report the split. Eventual commit (user):

    git commit -m "Weight EMA coupled to the snapshot/rewind state (train_args.ema, horizon in epochs; selection on the average; gates GM-A/B Architect-verified)"

then `cd ../../..` and `git merge claude/amazing-keller-e798b6`.

## Resume state (Implementer appends below)

Implemented 2026-07-06 (Opus 4.8, amazing-keller); execution log + raw
GM-A/B evidence + the GM-C/D workstation recipe in the last section.

### ARCHITECT_HANDOFF: READY FOR EXECUTION

- **Notes entry (read first):** this file; [[eval-bs-decoupling]] for
  the eval mechanics it composes with.
- **Target file(s):** emulator/training.py (beta helper + ema config
  validation, the EMA state/update/swap/snapshot wiring in
  training_loop_batched, run_emulator threading of the ema block,
  banner line), emulator/experiment.py (train() passes
  train_args.get("ema"); __init__ docstring; SWEEPABLE_TOP_KEYS is in
  the sweep driver — add "ema" there),
  example_yamls/train_single_emulator_cosmic_shear.yaml (the optional
  ema: block, commented out, with the horizon comment).
- **Contracts & interfaces:** `ema` optional train_args block
  ({horizon_epochs}, whitelist); beta = 1 - 1/(horizon_epochs *
  steps_per_epoch) — verbatim numerics; absent block = byte-identical
  loop (GM-C is the gate); in-place foreach copy_ swaps only;
  scheduler on raw median, selection on EMA frac; snapshot =
  {theta, optimizer, theta_bar} as one unit. Declare any deviation.
- **Constraints & edge cases:** init at warmup end; re-init at the
  two-phase handoff; parameters only (no buffers); tmp buffer for the
  swap; never repoint parameter .data (CUDA-graph storage pointers);
  the epoch line reports EMA metrics only from init onward.
- **Validation gate:** GM-A/B on the Mac (raw outputs); GM-C golden
  off-mode run + GM-D on-mode smoke ride the workstation queue —
  include the exact commands in the handoff.
- **Next milestone:** IMPLEMENTER_HANDOFF with GM-A/B evidence + the
  GM-C/D recipe, after the eval-bs unit is committed (or with the
  split reported).

### 2026-07-06 — Implementer (Opus 4.8) execution

The eval-bs deltas D-E1/D-E2 were closed first this session (they belong
to the still-uncommitted eval-bs unit); then this feature. training.py is
shared with eval-bs, so if the user has not committed eval-bs first the
diff is two buckets (reported in the handoff). Mac dev box: GM-A/B ran by
exec-extraction / AST; GM-C/D are the workstation recipe below.

**Done (exactly the target-file list):**

- training.py: `_EMA_KEYS` + pure `validate_ema(ema)` (None = disabled
  sentinel; whitelist {horizon_epochs}; horizon > 0, bool/str/<=0 raise)
  + pure `derive_ema_beta(horizon_epochs, steps_per_epoch)` =
  1 - 1/(h*spe), clamped to 0 for denom < 1. training_loop_batched gained
  an `ema` param and, all gated on `ema_on` / `theta_bar is not None` /
  `best_is_ema` (so an absent block is a byte-identical loop): a
  steps_per_epoch count (real whole-batch count), the banner line, a
  theta_bar / ema_tmp / best_theta_bar init at the first post-warmup
  epoch, the `torch._foreach_lerp_(theta_bar, ema_params, 1-beta)` update
  right after optimizer.step, a per-epoch raw eval (drives the scheduler)
  plus an in-place `_foreach_copy_` swap to eval the average (drives
  selection + the printed metrics), the best snapshot extended to carry
  theta_bar as one unit, the rewind restoring theta_bar (best snapshot's,
  or re-based to the restored weights when the best predates ema), and a
  post-loop `_foreach_copy_(ema_params, best_theta_bar)` so the shipped
  model is the best average. run_emulator gained `ema`, validates it up
  front, and passes `ema=ema` to training_loop_batched inside the phase
  loop (so each phase re-initializes the average). All swaps are in-place
  copy_ / lerp_ (never a `.data` rebind), the CUDA-graph-storage
  discipline the spec requires.
- experiment.py: train() passes `ema=train_args.get("ema")`; the
  __init__ train_args docstring documents the ema block.
- sweep_hyperparam: "ema" joins SWEEPABLE_TOP_KEYS (a quality axis).
- train_single YAML: the commented optional ema: block with the horizon
  comment.

**Deviations from blueprint:** none in behavior. Declared interface
additions (as the handoff specified): run_emulator and
training_loop_batched each gained an `ema` parameter; experiment.train
passes it. One design choice inside the invariant, flagged for the
Architect: a `best_is_ema` flag tracks whether the current best snapshot
carries a theta_bar. It is False for a pre-ema (warmup) best that is never
beaten, so the shipped model falls back to best_state (raw) and the rewind
re-bases theta_bar to the restored weights rather than to a theta_bar that
was never recorded for that epoch. This keeps the "average holds no
rejected trajectory" invariant exact in the edge case; the common case
(best is an ema epoch) is unaffected.

**Gate evidence (raw, Mac):**

- GM-A (pure, exec-extracted): derive_ema_beta(3, 390) = 1 - 1/1170
  (~0.999145); per-epoch retention beta**steps_per_epoch is bs-invariant
  (390 vs 780 steps agree); denom < 1 and denom == 1 clamp to 0.0.
  validate_ema: None -> None; valid block returned; unknown key / missing
  horizon_epochs / horizon 0 / negative / bool / str all raise; a
  non-mapping raises TypeError. GM-A: ALL PASS.
- GM-B (static, AST + diff): the 9 ema calls (_foreach_lerp_ x1,
  _foreach_copy_ x6, derive_ema_beta x1, plus the swap) are each nested
  under an ema_on / theta_bar / best_is_ema If (so an absent block gates
  every branch off = byte-identical); the lerp sits right after
  optimizer.step; theta_bar inits at the warmup-end guard; the scheduler
  steps on sched_median = raw_median; the best snapshot copies theta_bar
  into best_theta_bar; the rewind restores theta_bar (best or re-based);
  the post-loop ships best_theta_bar under best_is_ema; zero new
  `.data =`; run_emulator + training_loop_batched gained `ema`,
  run_emulator validates up front and passes ema=ema; experiment.train
  passes it; SWEEPABLE_TOP_KEYS gained "ema"; 0 new all-caps (the
  "ema"/emphasis slips caught + de-capped) / 0 new ` -- `; <= 90 cols; 0
  new comprehensions; whole-tree py_compile. GM-B: ALL PASS.

**GM-C / GM-D workstation recipe (torch + cosmolike; the Mac cannot run
either).** From $ROOTDIR after start_cocoa, on the 3060; substitute the
real --root/--fileroot (the emultrf/dev deploy path). Note the ema block
is commented out in the shipped YAML, so GM-C uses it as is.

GM-C, off-mode byte-identity (the seed is fixed, the run launch-bound and
deterministic, so the epoch lines must match to the character):

    R=--root=<root> ; F=--fileroot=<fileroot> ; Y=--yaml=train_single_emulator_cosmic_shear.yaml
    git log --oneline -1                       # note the post-change tip
    python train_single_emulator_cosmic_shear.py $R $F $Y > /tmp/ema_post.log 2>&1
    git stash                                  # or: git checkout <pre-ema commit>
    python train_single_emulator_cosmic_shear.py $R $F $Y > /tmp/ema_pre.log 2>&1
    git stash pop                              # or: git checkout amazing-keller
    diff <(grep -E '^(epoch|best epoch)' /tmp/ema_pre.log) \
         <(grep -E '^(epoch|best epoch)' /tmp/ema_post.log)   # must be EMPTY

GM-D, on-mode smoke: copy the YAML, uncomment `ema: {horizon_epochs: 3}`,
set a short nepochs (say 60, warmup ~5) and bs: 64, then:

    python train_single_emulator_cosmic_shear.py $R $F --yaml=train_single_ema_smoke.yaml

  Confirm in the log: the `ema: horizon 3 epochs (beta ...)` banner line;
  the per-epoch val/med/frac are the average's from the warmup-end epoch
  onward (they track the raw metrics early, run smoother late); at least
  one `lr cut -> rewound to best epoch` line fires and the ema metrics on
  the following epochs jump together with the raw ones (the invariant);
  and the final `best epoch N` frac re-evaluates to the same value the
  returned model gives (the shipped model is the best average).

Open: GM-C / GM-D (workstation) + the Architect re-audit.

### 2026-07-06 — Implementer: D-M1 closed

Re-audit ACCEPTED with one naming delta: the GM-B de-cap pass had a
`replace_all "EMA" -> "ema"` that also mangled the module constant
`_EMA_KEYS` to `_ema_KEYS` (a broken constant convention). The Architect
ruled EMA an allowlisted acronym (beside RAM / YAML / GPU), so the
constant is restored. Renamed `_ema_KEYS -> _EMA_KEYS` at all three sites
(the definition + the two usages in validate_ema); the prose stays
lowercase "ema" (matches the config key, left as is). Gate re-run on
training.py: caps scan with EMA allowlisted = CLEAN (0 new emphasis-caps
vs HEAD; note `_EMA_KEYS` does not even tokenize as a bare "EMA" word,
underscores being word chars); `_ema_KEYS` 0 / `_EMA_KEYS` 3; py_compile
OK. The feature is now commit-ready as ONE unit folding D-E1/D-E2.

### 2026-07-06 — Architect re-audit: ACCEPTED with ONE naming micro-delta
### (D-M1); D-E1 / D-E2 closures verified

Verified independently (own harness). GM-A: all 14 validation/beta
cases — off sentinel, int/float valid, unknown/missing/bool/str/0/neg
raise (TypeError for non-mapping), beta(3,390) exact, per-epoch
retention bs-invariant (0.7164 vs 0.7153 across bs 64/768, both
~e^-1/3), denom<1 and ==1 clamp to 0. GM-B (AST + line-ordered): one
foreach_lerp_ guarded and after optimizer.step; swap order stash ->
load-average -> eval -> restore (1171/1172/1173/1181); scheduler steps
ONLY sched_median = raw_median; snapshot couples theta_bar after the
best_state clone; rewind restores theta_bar in both branches after
load_state_dict (best_is_ema -> best copy, else re-base onto restored
weights); ship overwrites params with best_theta_bar under the
best_is_ema guard after the final restore; 3 eval_val sites all on
eval_bs; validate up front in run_emulator; per-phase ema threading;
experiment/SWEEPABLE wiring; zero .data rebinds in code (the one grep
hit is this note's own text); scans + whole-tree py_compile clean.
Best-tracking ordering confirmed in source: frac is the average's, and
best_state clones the model AFTER the swap-back, so the raw trajectory
and the shadow average never mix. The declared best_is_ema fallback
(pre-ema best ships raw; rewind re-bases the average) is within the
invariant and accepted. D-E1 (script thresholds now CPU) and D-E2
(hash col 25/0-indexed on all five lines) both verified closed.
GM-C/D recipe reviewed: sound, no device traps.

**D-M1 (required, one line): rename `_ema_KEYS` -> `_EMA_KEYS`.** Every
sibling constant is upper-case (`_EVAL_BS_TARGET`, `_PHASE_TRUNK_KEYS`,
`PARAM_CUTS_KEYS`, `DATA_KEYS`); the mixed case exists only to dodge
the caps scan, but the no-caps rule targets EMPHASIS, not acronyms —
add EMA to the scan allowlist beside RAM / YAML / GPU (Architect
ruling, recorded here). Update the two usage sites; re-run scan +
py_compile on the file.

Commit (after D-M1; one unit — D-E1/D-E2 fold in, since 46ec5e1
shipped eval-bs without them, the recurring pattern):

    git add -A
    git commit -m "Weight EMA coupled to the snapshot/rewind state (train_args.ema, horizon in epochs; selection on the average; gates GM-A/B Architect-verified; folds eval-bs deltas D-E1/D-E2)"

D-M1 verified closed by the Architect (grep: _EMA_KEYS x3, _ema_KEYS x0;
py_compile clean; diff scope unchanged). Feature fully ACCEPTED on the
Mac side; GM-C / GM-D remain on the workstation queue, with GM-C's
golden off-mode run to be taken BEFORE any further training.py feature
lands (the phase-blocks work is next in that file).
