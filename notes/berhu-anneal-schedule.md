---
name: berhu-anneal-schedule
description: "Design spec (Architect, 2026-07-06): an optional anneal sub-block inside loss.berhu — {hold_epochs, anneal_epochs, shape}, trim's argument names — that starts a berhu run as plain sqrt everywhere (the init mode) and cosine-blends into the full berhu/berhu_capped shape: L_s(c) = (1-s) sqrt(c) + s L_berhu(c), s = 0 through hold_epochs then shape-annealed 0 -> 1 over anneal_epochs. User's physics: early training is full of big outliers (chi2 > cap) passing DOWN through the (knot, cap) window — escalating window votes then wastes capacity on points that are merely en route; the window pressure belongs in late training when few points remain above the cap. The user's activation bool is realized as block PRESENCE (the ema/focus house pattern); absent = full berhu from epoch 1 = today's behavior, byte-identical (static per-pass specialization, no blend ops in the graph). s threads as a 0-dim device tensor updated in place per epoch (the trim_t/focus_t compile-safe pattern); both blend endpoints are C1, so every intermediate loss is C1 free."
metadata:
  node_type: memory
  type: project
---

# loss.berhu.anneal: sqrt -> berhu on a trim-style schedule

User request 2026-07-06: add hold_epochs / anneal_epochs / shape (+ an
activation switch) to the berhu block — "on hold epochs chi2 is just
sqrt(chi2) everywhere (init_mode); then a cosine transition between the
init mode and berhu_capped". Physics: in early training the (knot, cap)
window is polluted by big outliers (chi2 > cap) passing down through
it; escalated window votes belong in late training, when the population
above the cap is small.

## The math (verbatim numerics)

With s(epoch) in [0, 1] and L_mode the active berhu / berhu_capped
form (knots as before):

    L_s(c) = (1 - s) sqrt(c) + s L_mode(c)

    s(e) = 0                                    e <= hold_epochs
    s(e) = ramp((e - hold) / anneal_epochs)     hold < e <= hold+anneal
    s(e) = 1                                    afterwards

ramp = the SAME shapes the trim / focus schedules accept (cosine /
linear), evaluated by the SAME existing schedule helper (start = 0,
end = 1) — no new schedule code. Properties, all free:
- s = 0 -> exactly sqrt (0 * finite + 1 * v is exact in fp);
- s = 1 -> exactly the berhu form;
- every intermediate L_s is C1 in c (convex combination of two C1
  functions); the gradient-weight blend interpolates the vote profile
  smoothly from flat-1/2 to the windowed one;
- monotone in s at every c > knot (L_berhu >= sqrt there).

## Design

1. **Schema:** an optional `anneal:` sub-block inside loss.berhu, with
   trim's argument names:

       loss:
         mode: berhu_capped
         berhu:
           knot: 0.2
           cap:  10
           anneal:                # presence = on; absent = full berhu
                                  # from epoch 1 (today's behavior)
             hold_epochs:   50
             anneal_epochs: 300
             shape:         cosine

   The user's "bool to activate" is realized as block PRESENCE — the
   ema / focus house pattern (one less key; an `anneal: false` line is
   the block simply not written). Flagged as an Architect
   interpretation the user may veto in favor of an explicit bool key.
2. **Validation (extend validate_berhu):** berhu whitelist grows to
   {knot, cap, anneal}; the anneal sub-block whitelist is
   {hold_epochs, anneal_epochs, shape} with hold_epochs >= 0 int,
   anneal_epochs >= 1 int (bools rejected), shape one of the trim
   schedule's shapes; a malformed sub-block raises naming
   loss.berhu.anneal. Locality is free: anneal lives inside berhu,
   which already requires a berhu mode.
3. **Threading (the trim_t pattern):** when the anneal block is
   present, training_loop_batched keeps `s_t`, a 0-dim device tensor,
   filled in place at the top of each epoch from the existing schedule
   helper (a Python float in the traced closure risks recompiles /
   CUDA-graph loss; in-place tensor updates are the established
   pattern). _fwd_loss passes it as `berhu_s=s_t`; the _reduce berhu
   branches become v = (1 - berhu_s) * sqrt(c) + berhu_s * v_mode.
4. **Static specialization — absent block = byte-identical:** whether
   the anneal exists is fixed per pass (like the mode string), so the
   blend ops exist in the compiled graph ONLY when the block is
   present; without it the berhu branches are today's code verbatim
   (the golden gate).
5. **Per-phase semantics:** the loss block is already per-phase full
   replacement; a phase's anneal schedule counts from that pass's own
   epoch 1 (the trim / focus restart rule, same wording in the docs).
6. **Banner:** `loss_mode berhu_capped (knot 0.2, cap 10; anneal:
   hold 50 + 300 cosine)` when present.
7. **Docs:** loss_functions.py _reduce/loss docstrings (berhu_s kwarg,
   the blend); training.py run_emulator loss doc + validate_berhu
   docstring; experiment.__init__; train_single YAML (the commented
   block above); tune comment (anneal ranges are legitimate search
   leaves: loss.berhu.anneal.hold_epochs). Paste every changed YAML
   block.
8. **Advice recorded, not enforced:** set hold_epochs at least as long
   as trim's hold, so the trim machinery absorbs monsters while the
   berhu is still soft; the two schedules compose per-sample and are
   deliberately independent knobs.

## Validation gate

- GBA-A (pure math, Mac, numpy): s = 0 reproduces sqrt exactly and
  s = 1 the berhu form exactly (both modes, default + non-default
  knots); intermediate s = 0.3: L_s == (1-s) sqrt + s L_mode
  pointwise, C1 at both knots (value + slope straddles), monotone in
  s at c > knot; the schedule: s(e) == the trim helper's value with
  start 0 / end 1 for cosine and linear, hold boundary exact.
- GBA-A2 (pure config, Mac): whitelist {knot, cap, anneal}; sub-block
  whitelist + value checks (hold 0 ok, negative/bool/str raise;
  anneal_epochs 0 raises; bad shape raises naming the allowed);
  absent sub-block resolves to None (off).
- GBA-B (static, Mac): blend ops present ONLY under the
  anneal-present specialization (grep/AST: the no-anneal berhu branch
  is textually today's); s_t is a 0-dim device tensor filled in place
  per epoch by the existing schedule helper (no new schedule code);
  per-pass restart; banner; docs; scans; whole-tree py_compile.
- GBA-C (workstation, rides the queue): golden — a berhu run WITHOUT
  the anneal block is byte-identical pre/post; smoke — a short run
  with {hold 5, anneal 10, cosine}: the printed loss is continuous at
  the hold boundary, s reaches 1 by epoch 15 (one debug line or the
  banner), and early-epoch behavior matches a plain-sqrt run.

## Sequencing

Serialize behind the loss-block-nesting commit (same berhu machinery).
Eventual commit (user):

    git commit -m "Anneal the berhu loss from sqrt on a trim-style schedule (loss.berhu.anneal {hold_epochs, anneal_epochs, shape}; C1 blend, graph-safe s tensor; gates GBA-A/A2/B Architect-verified)"

then `cd ../../..` and `git merge claude/amazing-keller-e798b6`.

## Resume state (Implementer appends below)

### 2026-07-06 — Implementer (Opus 4.8) execution

Clean base: loss-block-nesting committed `dcbaf9f`. This unit adds the
optional anneal schedule on top of the nested loss block.

**Done (the full target-file list):**

- training.py: `validate_berhu` whitelist grows to {knot, cap, anneal}; it
  now always returns {knot, cap, anneal} (anneal the validated sub-block or
  None). New pure `_validate_berhu_anneal(anneal, which)` + `_ANNEAL_KEYS`
  (hold_epochs, anneal_epochs, shape) + `_ANNEAL_SHAPES` (const, linear,
  cosine, step): mapping-only; whitelist; all three keys required;
  hold_epochs >= 0 int, anneal_epochs >= 1 int (bool rejected); shape in the
  set. training_loop_batched builds `s_t` (None when no anneal, else a
  0-dim device tensor) + `s_opts` = {start 0, end 1, **anneal} beside
  knot_t/cap_t; fills s_t in place per epoch via `anneal_value` (the
  existing helper, no new schedule code) beside trim_t/focus_t; `_fwd_loss`
  gains a `berhu_s` parameter passed in BOTH loss() calls, threaded from
  the call site `fwd_loss(..., trim_t, focus_t, s_t)` (the trim_t/focus_t
  arg pattern). Banner appends "; anneal: hold H + A shape" when present.
  run_emulator loss doc + training_loop_batched berhu doc updated.
- loss_functions.py: `berhu_s=None` on `CosmolikeChi2.loss` + `_reduce`
  (forwarded); each berhu branch keeps today's `torch.where(...)` form
  verbatim, then adds, guarded by `if berhu_s is not None:`, the blend
  `v = (1.0 - berhu_s) * torch.sqrt(c) + berhu_s * v` (C1 for every s).
  Docstrings updated. The blend ops enter the compiled graph only when
  berhu_s is passed (static per-pass specialization -> a no-anneal run is
  byte-identical). RescaledChi2 / IA losses forward via *args/**kwargs
  (berhu_s flows free); ElementWeightedChi2 is untouched (no berhu support).
- experiment.py: __init__ loss doc gains the anneal sub-block.
- YAMLs: train_single commented anneal: sub-block inside loss.berhu; tune
  loss.berhu.anneal.{hold_epochs, anneal_epochs} search-leaf comment.

**Deviations (FLAGGED, doc-coherence beyond the target list):** the two IA
forwarding docstrings (`emulator/IA/loss_functions.py`) enumerate the
forwarded kwargs; added `berhu_s` alongside `berhu_knot, berhu_cap`. No IA
code change.

**Interface (declared):** loss()/_reduce gained `berhu_s`; _fwd_loss gained
a positional berhu_s (internal). training_loop_batched/run_emulator
signatures unchanged (the anneal rides inside the existing `berhu` dict /
`loss` block). Edge cases confirmed legal + documented: hold_epochs 0 (ramp
from epoch 1) and anneal_epochs beyond nepochs (s never reaches 1).

**Gate evidence (raw, Mac — no torch: GBA-A is numpy math, GBA-A2/B
exec-extract + static):**

    === GBA-A  blend + schedule math ===   (all OK)
      3 knot pairs {(0.2,10),(0.05,1),(1,50)}: s=0 == sqrt exactly, s=1 ==
      the mode form exactly (berhu + capped); s=0.3 == (1-s)sqrt + s*mode
      pointwise; monotone non-decreasing in s for c>knot; value + slope C1
      straddles at BOTH knots. Schedule (anneal_value, start 0/end 1):
      cosine + linear hold-boundary exact (s(50)=0, s(51)>0, s(350)=1),
      monotone; hold 0 ramps from epoch 1; anneal>>nepochs keeps s<1.
    === GBA-A2  validate_berhu / _validate_berhu_anneal ===   (13/13 OK)
      resolved berhu carries anneal (None / validated); whitelist grew;
      hold 0 ok; hold negative/bool, anneal_epochs 0/str, bad shape,
      missing key, non-mapping, anneal-on-non-berhu-mode all raise.
    === GBA-B  static ===   (14/14 OK)
      blend present exactly twice, each only under `if berhu_s is not
      None:`; both where() forms textually today's; berhu_s on loss()/
      _reduce; s_t 0-dim device tensor filled in place from anneal_value
      (no new schedule code); both _fwd_loss calls + the 5-arg call site;
      whitelist + constants; banner note.

    GBA gate: ALL PASS

    House scans (diff vs HEAD): 0 over-width, 0 new ` -- `, new all-caps =
    {IA, YAML} (allowlisted acronyms). Whole-tree py_compile OK.

**GBA-C (workstation, rides the queue).** Two legs from $ROOTDIR:

    R=--root=<root> ; F=--fileroot=<fileroot>
    Y=--yaml=train_single_emulator_cosmic_shear.yaml
    # leg 1 golden: a berhu run WITHOUT the anneal block is byte-identical
    # pre/post this feature. Use a loss: {mode: berhu_capped, berhu:
    # {knot: 0.2, cap: 10}} config (NO anneal key).
    git stash && python train_single_emulator_cosmic_shear.py $R $F $Y \
      > /tmp/ba_pre.log 2>&1
    git stash pop && python train_single_emulator_cosmic_shear.py $R $F $Y \
      > /tmp/ba_post.log 2>&1
    diff <(grep -E '^(phase|epoch|best)' /tmp/ba_pre.log) \
         <(grep -E '^(phase|epoch|best)' /tmp/ba_post.log)   # EMPTY
    # leg 2 smoke: a short run with the anneal on,
    #   loss:
    #     mode: berhu_capped
    #     berhu: {knot: 0.2, cap: 10, anneal: {hold_epochs: 5,
    #                                          anneal_epochs: 10,
    #                                          shape: cosine}}
    # banner shows "...(knot 0.2, cap 10; anneal: hold 5 + 10 cosine)";
    # the printed train loss is CONTINUOUS at the hold boundary (epoch 5->6,
    # s leaves 0 smoothly under cosine); by epoch 15 s = 1 (full berhu); the
    # first ~5 epochs match a plain loss: {mode: sqrt} run (s = 0 = sqrt).

**Commit (user-side).** One clean unit; the command is in Sequencing above.
Open: GBA-C (workstation) + the Architect re-audit.

### ARCHITECT_HANDOFF: READY FOR EXECUTION

- **Notes entry (read first):** this file + [[loss-block-nesting]] +
  [[loss-mode-berhu]]. Serialize behind the loss-block-nesting commit.
- **Target file(s):** emulator/training.py (validate_berhu whitelist +
  anneal sub-block validation; s_t threading beside knot_t/cap_t with
  the per-epoch in-place fill from the EXISTING schedule helper;
  banner), emulator/loss_functions.py (berhu_s kwarg through
  loss()/_reduce; the blend in the two berhu branches under the
  anneal-present specialization only), emulator/experiment.py
  (docstring), example_yamls train_single + tune (comments; paste
  final blocks).
- **Contracts & interfaces (verbatim numerics):** the blend and
  schedule forms above; reuse the trim/focus schedule evaluator with
  start = 0, end = 1 (no new schedule code); absent anneal block =
  today's code paths byte-identical (GBA-C golden); s_t never a
  closure float. Declare any deviation.
- **Constraints & edge cases:** hold_epochs = 0 (ramp from epoch 1)
  and anneal beyond nepochs (s never reaches 1 — allowed, document)
  both legal; per-phase restart at the pass's epoch 1; non-berhu
  modes untouched.
- **Validation gate:** GBA-A/A2/B on the Mac (raw outputs); GBA-C
  rides the workstation queue — write the recipe now and embed it.
- **Next milestone:** IMPLEMENTER_HANDOFF with GBA-A/A2/B evidence +
  the embedded GBA-C recipe.

### 2026-07-06 — Architect re-audit: ACCEPTED (no deltas; one user
### confirmation outstanding)

Verified independently (own harness, 33 checks, all PASS). GBA-A: the
schedule holds 0 through the hold, hits exactly 1 at hold+span, cosine
monotone with exact midpoint 0.5, linear exact; the blend reproduces
sqrt exactly at s=0 and the berhu form exactly at s=1 (two knot pairs,
array_equal); C1 straddles at BOTH knots under s=0.3; monotone in s
beyond the knot. GBA-A2: _ANNEAL_KEYS/_ANNEAL_SHAPES mirror the shared
anneal_value helper exactly (const/linear/cosine/step — faithful
reuse; note const with start 0 pins s=0 forever, a legal odd choice
inheriting the helper's documented semantics); all three keys
required; the nine rejection cases; the anneal threads through
validate_loss and stays impossible under a non-berhu mode (locality
inherited). GBA-B: the blend is guarded in exactly the two berhu
branches and nowhere else; s_t is a 0-dim tensor filled in place in
the epoch loop beside trim_t/focus_t with {start 0, end 1} into the
EXISTING helper (no new schedule code); berhu_s stays None when the
block is absent (the static specialization the golden gate rests on);
scans + whole-tree py_compile clean.

Outstanding (not a delta): the activation-by-block-PRESENCE
interpretation of the user's bool — flagged by both Architect and
Implementer; the user confirms or vetoes (a veto = a small delta
adding an explicit key). GBA-C rides the workstation queue.

Commit (user, after confirming the presence semantics):

    git add -A
    git commit -m "Anneal the berhu loss from sqrt on a trim-style schedule (loss.berhu.anneal {hold_epochs, anneal_epochs, shape}; C1 blend, graph-safe s tensor; gates GBA-A/A2/B Architect-verified)"
