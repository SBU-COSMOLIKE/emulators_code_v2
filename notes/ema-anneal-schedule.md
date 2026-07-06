---
name: ema-anneal-schedule
description: "Design spec (Architect, 2026-07-06): an optional anneal sub-block inside train_args.ema — {hold_epochs, anneal_epochs, shape}, the berhu-anneal twins — that defers and ramps the weight-average's MEMORY instead of starting it at full horizon from warmup end. Mechanism: schedule the horizon, not a blend — h(e) = horizon_epochs * s(e) with s the trim-style 0->1 schedule, and per epoch beta(e) = derive_ema_beta(h(e), steps_per_epoch); s=0 gives beta=0 (theta_bar tracks theta exactly, zero memory of the terrible era — the existing denom<1 clamp already implements it), and the memory grows continuously to the target as s ramps. User's physics: a fixed 3-epoch memory started at warmup end still averages in steeply-improving (bad) weights all through mid-training; full smoothing belongs where the trajectory is flat. Key implementation note: beta here may be a per-epoch Python float — the EMA lerp is EAGER (outside every compiled graph), unlike trim_t/focus_t/s_t which feed compiled code; document the distinction. EMA goes live at max(warmup end, first s>0); per-pass restart; absent block = today's behavior byte-identical."
metadata:
  node_type: memory
  type: project
---

# ema.anneal: ramp the average's memory, not just its start

User request 2026-07-06 (after the berhu anneal): "none of this can
start in the beginning where the weights are terrible". Agreed, with a
sharpening: the current warmup-end start dodges the lr thrash, but the
steep-descent era lasts far longer than warmup — a full-horizon
average there lags the model by ~a horizon (one horizon back keeps
~37% weight) and pollutes mid-training smoothing.

## The math (verbatim numerics; both helpers already exist)

    s(e)    = anneal_value(e, {start: 0, end: 1, hold_epochs,
                               anneal_epochs, shape})
    h(e)    = horizon_epochs * s(e)
    beta(e) = derive_ema_beta(h(e), steps_per_epoch)
              (= 0 whenever h(e)*steps_per_epoch < 1 — the existing
               clamp; theta_bar then tracks theta exactly, i.e. the
               average has no memory during the hold)

Properties, all free: continuous everywhere (no init discontinuity —
at the moment beta first exceeds 0, theta_bar == theta); the memory
length IS h(e), so the terrible era decays out with the short early
time-constant; after the ramp, beta equals the no-anneal value exactly
(the no-anneal run is the s == 1 special case).

## Design

1. **Schema** (the berhu-anneal twins, same names, same validator
   semantics):

       ema:
         horizon_epochs: 3
         anneal:                # presence = on; absent = today's
                                # behavior (full horizon from warmup
                                # end), byte-identical
           hold_epochs:   50
           anneal_epochs: 300
           shape:         cosine

2. **Validation:** _EMA_KEYS grows to {horizon_epochs, anneal}; the
   anneal sub-block reuses the SAME validator as the berhu anneal —
   generalize `_validate_berhu_anneal` into one shared
   `_validate_anneal_block(anneal, which)` (it is already
   berhu-agnostic in substance; the berhu call sites keep their
   behavior — regression-gated).
3. **Live point:** the EMA allocates (theta_bar <- theta) at the
   LATER of warmup end (the existing floor, unchanged rationale) and
   the first epoch with s > 0; before that it is dormant exactly as
   the pre-init state is today (raw eval/selection). From the live
   point, beta follows the schedule.
4. **beta is a per-epoch Python float — deliberately.** The EMA lerp
   runs EAGER in the epoch loop (theta_bar is a private buffer, never
   graph-captured), so the trim_t/focus_t/s_t tensor discipline does
   NOT apply; recompute beta(e) at the top of each epoch beside the
   other anneal_value calls and pass it to the (eager)
   _foreach_lerp_. Document the distinction in a comment — the next
   reader must not "fix" it into a tensor, nor copy this float
   pattern into compiled-side schedules.
5. **Per-phase:** the schedule restarts at each pass's epoch 1
   (trim/focus/berhu-anneal rule); theta_bar already re-inits per
   pass.
6. **Rewind:** unchanged — theta_bar restores with the snapshot;
   schedules do not rewind (they are functions of the epoch counter,
   like trim/focus).
7. **Banner:** `ema: horizon 3 epochs (beta -> 0.99915; anneal:
   hold 50 + 300 cosine; selection + metrics on the average,
   scheduler on the raw median)` — the target beta, arrow-marked as
   the ramp endpoint.
8. **Docs:** experiment.__init__ ema entry; train_single YAML
   commented block (the twins pattern beside berhu's); the
   weight-ema note gains a pointer here.

## Validation gate

- GME-A (pure, Mac): h(e) = target * s(e) through both existing
  helpers — beta == 0 through the hold and while h(e)*steps < 1;
  ramp-end beta == the no-anneal beta exactly; cosine midpoint;
  the clamp boundary epoch (first beta > 0) is continuous
  (theta_bar == theta there by construction); linear + cosine shapes.
- GME-A2 (pure config, Mac): _EMA_KEYS {horizon_epochs, anneal};
  the shared validator generalization — all nine berhu-anneal
  rejection cases re-run against BOTH call paths (ema.anneal and
  loss.berhu.anneal error texts name their own path); absent
  sub-block -> None.
- GME-B (static, Mac): live point = max(warmup end, first s > 0);
  beta recomputed per epoch beside the anneal_value calls, passed to
  the EAGER lerp (the float-vs-tensor comment present); absent block
  -> the beta computation is the existing one-time constant
  (byte-identical, guard-gated); per-pass restart; banner; scans;
  whole-tree py_compile.
- GME-C (workstation, rides the queue): golden — an ema run WITHOUT
  anneal is byte-identical pre/post; smoke — {hold 5, anneal 10}:
  EMA metrics appear at the live point, banner shows the schedule,
  and the printed metrics converge to the raw ones' smoothed
  neighborhood as s -> 1.

## Sequencing

Serialize behind the pending D-P2v2 unit (same training.py). Eventual
commit (user):

    git commit -m "Anneal the EMA horizon on a trim-style schedule (ema.anneal {hold_epochs, anneal_epochs, shape}; beta(e) via the existing helpers, eager-float by design; gates GME-A/A2/B Architect-verified)"

then `cd ../../..` and `git merge claude/amazing-keller-e798b6`.

## Resume state (Implementer appends below)

(none yet)

### ARCHITECT_HANDOFF: READY FOR EXECUTION (after the D-P2v2 unit)

- **Notes entry (read first):** this file + [[weight-ema-snapshot-coupled]]
  + [[berhu-anneal-schedule]] (the validator to generalize).
- **Target file(s):** emulator/training.py (_EMA_KEYS + the shared
  anneal-block validator generalization; the per-epoch beta(e) beside
  the anneal_value calls; live-point rule; banner),
  emulator/experiment.py (docstring), train_single YAML (the
  commented block; paste final form). The banner rendering follows
  [[banner-prints-consumed-view]] (state it in resolved form — this
  spec's item 7 is that statement).
- **Contracts (verbatim numerics):** h(e) = horizon_epochs * s(e);
  beta(e) = derive_ema_beta(h(e), steps_per_epoch); s from
  anneal_value with start 0 / end 1; absent block byte-identical;
  beta stays an eager Python float WITH the explanatory comment;
  the shared validator keeps both call paths' error texts
  path-correct. Declare any deviation.
- **Validation gate:** GME-A/A2/B on the Mac (raw outputs); GME-C
  rides the workstation queue — embed the recipe.
- **Next milestone:** IMPLEMENTER_HANDOFF with GME-A/A2/B evidence +
  the embedded GME-C recipe.

### 2026-07-06 — AMENDED pre-execution (user-found gap): ema joins the
### phase blocks

User report: `trunk: ema: {...}` rejected — "unknown train_args.trunk
key(s): ['ema']". The phase whitelist never got ema (it predates the
phase machinery by one commit), violating the mirror-the-top-level
principle. The user's physics is the design argument: the two passes
are independent trainings (fresh optimizer / scheduler / warmup, and
theta_bar ALREADY re-initializes per pass), so trunk-only ema,
head-only ema, or different horizons per phase are all legitimate.

Design (folds into this unit):
1. "ema" joins _PHASE_BLOCK_KEYS (eight keys again), FULL REPLACEMENT
   like scheduler/trim/focus/loss: a phase ema block replaces the
   top-level one for that pass; absent in the phase = inherit the
   top level.
2. **Opt-out switch:** a phase may write `ema: null` (the key present
   with an empty value) to DISABLE an inherited top-level ema for
   that pass — validate_ema(None) is already the off sentinel; the
   resolution distinguishes key-present-None (off) from key-absent
   (inherit). Documented in the YAML comment.
3. Per-pass resolution beside loss_pass: ema_pass = the phase's block
   if the key is present (possibly None) else the top-level block;
   validate_ema per pass with which = "trunk.ema" / "head.ema" (error
   paths name the owner); training_loop_batched receives ema_pass.
4. Demotion (single-phase): automatic once the key is whitelisted —
   prefix-strip full-replaces, including trunk: ema: null -> top-level
   off.
5. validate_sweep_paths: generic strip covers trunk.ema.horizon_epochs
   -> 'ema.horizon_epochs' with no change.
6. Banner (the consumed-view directive): each phase line shows ITS
   resolved ema (horizon + anneal schedule, or nothing when off).

Gate additions (GMP):
- GMP-A (pure): trunk-only ema (the user's exact shape) -> trunk pass
  carries it, head pass none; head-only vice versa; different horizons
  per phase; ema: null disables an inherited block; single-phase
  demotion merges trunk.ema (incl. the null -> off edge); error paths
  name trunk.ema / head.ema.
- GMP-B (static): "ema" in _PHASE_BLOCK_KEYS (8); ema_pass resolution
  beside loss_pass; per-pass banner; the up-front top-level
  validate_ema stays (absent-phase inheritance validates once).
