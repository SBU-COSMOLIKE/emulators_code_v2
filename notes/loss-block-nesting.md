---
name: loss-block-nesting
description: "Design spec (Architect, 2026-07-06): nest all loss options under one train_args.loss block — loss: {mode: sqrt|chi2|sqrt_dchi2|berhu|berhu_capped, berhu: {knot, cap}} — replacing the flat loss_mode string + the top-level berhu block (user directive: loss-related options belong localized in a loss block, like model:/lr:/scheduler:). Payoffs beyond style: (1) the D-B1 inheritance machinery is DELETED — with the knots inside the same block as their mode, 'berhu sub-block without a berhu mode' is a purely LOCAL error (no run-level inertness guard, no phase-owned/inherited split); (2) the mode string gets validated EARLY for the first time (a typo'd mode used to explode at the first compiled loss call); (3) the phase whitelist shrinks 8 -> 7 ('loss' replaces 'loss_mode' + 'berhu', full replacement like scheduler/trim/focus). Loud no-alias migration for flat loss_mode:/berhu: (top level AND phase blocks), paste-ready blocks carrying values over. loss: absent = {mode: sqrt}."
metadata:
  node_type: memory
  type: project
---

# train_args.loss: one nested block for mode + berhu knots

User directive 2026-07-06: "options that are related to loss are within
the loss block - localized". The flat pair

    loss_mode: sqrt
    berhu: {knot: 0.2, cap: 10}

becomes

    loss:
      mode: berhu_capped
      berhu:
        knot: 0.2
        cap:  10

and plain sqrt is just `loss: {mode: sqrt}` — or no block at all
(absent = mode sqrt, the current default).

## Why this is better than the schema it replaces (not just style)

1. **D-B1 dissolves.** The inheritance patch (run-level inertness
   guard + the phase-owned vs inherited validation split in
   run_emulator) existed only because the knots lived apart from
   their mode. Nested, "a berhu: sub-block under a non-berhu mode:"
   is a LOCAL error checked inside one block — the guard and the
   split are DELETED, not adapted. The mixed config becomes
   self-documenting: the head's loss block carries its own knots.
2. **Early mode validation, free.** validate_loss whitelists the five
   mode strings at config time; today a typo'd loss_mode survives
   until the first compiled loss call raises mid-run.
3. **Phase whitelist shrinks 8 -> 7**: "loss" (full replacement, the
   scheduler/trim/focus semantics) replaces "loss_mode" + "berhu".

## Design

1. **Schema:** optional `train_args.loss` block, whitelist
   {mode, berhu}. `mode` in {sqrt, chi2, sqrt_dchi2, berhu,
   berhu_capped} (default sqrt; absent block = {mode: sqrt}).
   `berhu` = the existing {knot, cap} sub-block, valid ONLY when mode
   is berhu/berhu_capped (local check), validated by the existing
   validate_berhu(berhu, mode, which) — the loss_mode=None hook
   becomes dead and is removed; cap-under-plain-berhu stays accepted
   (sweep-friendliness).
2. **Pure `validate_loss(block, which)` in training.py** (beside
   validate_ema): None -> {"mode": "sqrt"}; non-mapping -> TypeError;
   unknown key / unknown mode -> ValueError naming the five; the
   berhu sub-block routed through validate_berhu with THIS block's
   mode; returns the resolved {"mode", "berhu"(resolved or None)}.
3. **Loud migration** (the param_cuts / phase-lr precedent, no
   aliases): a flat `loss_mode` or flat `berhu` key — top level OR
   inside a trunk:/head: block — raises a ValueError whose body is
   the paste-ready nested loss: block with the offending values
   carried over. Checked in validate_loss's caller context
   (run_emulator up front + resolve_phase_args before demotion, so
   errors are identical on both paths) and in validate_phase_block
   (whose whitelist now rejects loss_mode/berhu with the migration
   message, not a bare unknown-key error).
4. **Threading:** run_emulator's `loss_mode` and `berhu` parameters
   are REPLACED by one `loss=None`; per pass the resolved loss block
   is phase-full-replaced then validate_loss'd; mode_pass =
   loss_pass["mode"], knots from loss_pass["berhu"] (knot_t/cap_t as
   today, kappa_t discipline). experiment.train passes
   loss=train_args.get("loss"). The internal loss()/_reduce
   interfaces (mode, berhu_knot, berhu_cap kwargs) are UNCHANGED —
   this is a config-layer change.
5. **Phase machinery:** _PHASE_BLOCK_KEYS: -loss_mode, -berhu, +loss
   (7 keys; full replacement — a phase wanting berhu restates its
   whole loss block, knots included; explicit beats inherited).
   resolve_phase_args: prefix-strip covers "loss" with no code
   change. validate_sweep_paths: generic strip covers
   trunk.loss.mode -> 'loss.mode'.
6. **Sweeps:** SWEEPABLE_TOP_KEYS: -"loss_mode", -"berhu", +"loss"
   (dotted axes: loss.mode as a string sweep, loss.berhu.knot /
   loss.berhu.cap as numeric axes). Sweep/tune YAML comments updated.
7. **Banner:** unchanged output (`loss_mode berhu_capped (knot 0.2,
   cap 10)` reads from the pass's resolved loss block).
8. **Docs sweep:** run_emulator docstring; experiment.__init__
   train_args docstring; the three example YAMLs (loss: block with
   the five modes listed + the commented berhu sub-block + the head
   example); tune string-keys comment (loss.mode). Paste every
   changed YAML block. Grep gate: flat `loss_mode:` and top-level
   `berhu:` appear nowhere in py/yaml/README afterward (notes
   exempt).

## Migration examples (paste-ready)

Top level:

    loss:
      mode: sqrt              # sqrt | chi2 | sqrt_dchi2 | berhu
                              # | berhu_capped (absent block = sqrt)

Berhu head on a sqrt run (the production shape; the knots now travel
WITH the mode that uses them):

    loss:
      mode: sqrt
    head:
      loss:
        mode: berhu_capped
        berhu:
          knot: 0.2
          cap:  10

## Validation gate

- GL-A (pure, Mac): validate_loss — absent -> {mode: sqrt}; each of
  the five modes; unknown mode / unknown key loud (naming the five);
  berhu sub-block + non-berhu mode loud (LOCAL); berhu values still
  validated (knot >= cap etc. via validate_berhu); non-mapping
  TypeError; the dead None-mode hook removed from validate_berhu.
- GL-B (pure, Mac): migration — flat loss_mode (top + phase), flat
  berhu (top + phase), and both together each raise with the
  paste-ready nested block carrying the values; identical errors on
  the run_emulator and resolve_phase_args paths.
- GL-C (static, Mac): run_emulator signature has loss= and NO
  loss_mode=/berhu=; the D-B1 machinery (run-level inertness guard +
  phase_owns_berhu split) is GONE; per-pass resolution =
  full-replace + validate_loss; _PHASE_BLOCK_KEYS == 7 with "loss";
  SWEEPABLE swap; knot threading unchanged (kappa_t discipline, both
  _fwd_loss calls); banner; grep gate (no flat keys in py/yaml/
  README); house scans; whole-tree py_compile.
- GL-D (workstation, rides the queue): golden equivalence — the same
  physical config expressed in the new schema reproduces the
  pre-change run's epoch lines (config-layer change, numerics
  untouched); the berhu GB-C leg-2 recipe is UPDATED in
  notes/loss-mode-berhu.md to the new schema and runs as part of
  this gate.

## Sequencing

Serialize behind the berhu commit (this restructures what that unit
introduced — commit berhu first, then this lands as its own unit; if
berhu is still uncommitted the Implementer reports the split).
Eventual commit (user):

    git commit -m "Nest loss options under train_args.loss (mode + berhu knots localized; D-B1 inheritance machinery deleted; early mode validation; gates GL-A-C Architect-verified)"

then `cd ../../..` and `git merge claude/amazing-keller-e798b6`.

## Resume state (Implementer appends below)

### 2026-07-06 — Implementer (Opus 4.8) execution

Clean base: the berhu unit committed `64786a7` (D-B1 folded in). This unit
restructures that schema; it lands as its own commit.

**Done (the full target-file list):**

- training.py: pure `validate_loss(loss, which)` beside validate_ema
  (None -> {mode: sqrt, berhu: None}; whitelist {mode, berhu}; mode
  whitelist `_LOSS_MODES` = (chi2, sqrt, sqrt_dchi2, berhu, berhu_capped);
  routes `loss["berhu"]` through validate_berhu with `which_berhu` =
  "loss" / "{phase}.loss" so messages render train_args.loss.berhu;
  returns berhu = the resolved {knot, cap} for a berhu mode, else None).
  `validate_berhu`'s dead `loss_mode is not None` hook removed (now
  unconditional `if not is_berhu`); its messages reworded to "sub-block" /
  "mode". `_loss_migration_message(loss_mode, berhu, which)` builds the
  paste-ready nested block. `validate_phase_block` rejects flat
  loss_mode/berhu with that migration message and validates its `loss`
  sub-block. `_PHASE_BLOCK_KEYS` = 7 ("loss" replaces "loss_mode"+"berhu").
  run_emulator: signature `loss=None` replaces `loss_mode=`/`berhu=`; the
  D-B1 run-level inertness guard and the phase_owns_berhu split are DELETED
  and replaced by one up-front `validate_loss(loss, "train_args")` plus a
  per-pass full-replace + validate_loss (`mode_pass` = loss_pass["mode"],
  `berhu_pass` = loss_pass["berhu"]). knot_t/cap_t threading, both
  _fwd_loss loss() calls, and the banner ("loss_mode {mode}" text kept) are
  UNCHANGED. loss()/_reduce internals UNCHANGED.
- experiment.py: train() passes `loss=train_args.get("loss")` (loss_mode=/
  berhu= dropped); resolve_phase_args rejects a flat top-level loss_mode/
  berhu before its early return (so a two-phase model is covered too, since
  experiment reads only train_args["loss"]); banner reads
  `(loss or {}).get("mode")`; __init__ docstring rewritten to the loss
  block; imports validate_loss + _loss_migration_message.
- sweep driver: SWEEPABLE_TOP_KEYS -loss_mode -berhu +loss.
- YAMLs: train_single (loss: block + commented berhu sub-block + trunk/head
  loss examples + phase-whitelist comment), tune (loss.mode string-key
  comment + loss: block), sweep (loss: block). README loss_mode -> the
  nested loss: block.
- notes/loss-mode-berhu.md: GB-C leg-2 recipe updated to the new schema
  (the head owns its loss block; the D-B1 inheritance shape is gone).

**Deviations from the target-file list (FLAGGED, doc-coherence only):**
`loss_functions.py` had stale config paths `train_args.berhu.knot/.cap`
(2 loss()/_reduce docstrings + the 2 runtime None-guard messages) — updated
to `train_args.loss.berhu.*`; the message prose `loss_mode 'berhu'` ->
`mode 'berhu'`. `_reduce`/`loss()` code branches + kwargs UNCHANGED. Also
the train_single driver header comment `loss_mode` -> `loss`. Both are
pure doc/message text; left stale they would be exactly the doc-vs-code
drift the package audit gates on.

**Interface change (declared):** run_emulator loses `loss_mode`/`berhu`
params and gains `loss`; experiment.train's call updated in lockstep.
training_loop_batched's `berhu` param is UNCHANGED (run_emulator still
passes it `berhu=berhu_pass`).

**Gate evidence (raw, Mac — exec-extract, no torch):**

    === GL-A  validate_loss ===   (14/14 OK)
      absent -> {mode: sqrt, berhu: None}; chi2/sqrt/sqrt_dchi2 -> berhu
      None; mode berhu (no sub) -> knots default 0.2/10; berhu_capped +
      explicit knots honored; cap accepted under plain berhu; LOCAL
      berhu-sub-on-non-berhu raises (names train_args.loss + sub-block +
      no-op); unknown mode names the five; unknown key; non-mapping
      TypeError; knot>=cap via validate_berhu; phase which ->
      train_args.head.loss in the error.
    === GL-B  migration (both paths) ===   (10/10 OK)
      top-level flat loss_mode / berhu / both -> paste-ready block carrying
      the values; two-phase model still catches a top-level flat key;
      phase-block flat loss_mode / berhu -> migration (not unknown-key);
      single-phase demotion raises the SAME validate_phase_block message
      (== the direct call).
    === GL-C  static ===   (12/12 OK)
      run_emulator has loss= and NO loss_mode=/berhu=; phase_owns_berhu +
      "no pass runs a berhu mode" GONE; per-pass validate_loss(loss_raw...);
      up-front validate_loss(loss,"train_args"); _PHASE_BLOCK_KEYS 7 with
      loss (no loss_mode/berhu); SWEEPABLE swap; knot_t/cap_t from berhu_r;
      berhu_knot=knot_t in both _fwd_loss calls; banner keeps loss_mode
      text; validate_berhu None-hook removed.

    GL gate: ALL PASS

    House scans (diff vs HEAD): 0 lines > 90 cols; 0 new ` -- `; new
    all-caps in added .py/.yaml = {YAML} (allowlisted). Grep gate: no flat
    `loss_mode:` / top-level `berhu:` in py/yaml/README (notes exempt).
    Whole-tree py_compile OK.

**GL-D (workstation, rides the queue):** golden equivalence — the same
physical config in the new loss: schema reproduces the pre-change run's
epoch lines (config-layer change; numerics untouched). The berhu GB-C
leg-2 recipe (now the head-owns-loss shape) runs as part of this gate; see
[[loss-mode-berhu]].

**Commit (user-side).** One clean unit; the command is in Sequencing
above. Open: GL-D (workstation) + the Architect re-audit.

### ARCHITECT_HANDOFF: READY FOR EXECUTION

- **Notes entry (read first):** this file + [[loss-mode-berhu]] (whose
  GB-C leg-2 recipe this updates). Serialize behind the berhu commit.
- **Target file(s):** emulator/training.py (validate_loss + migration
  messages; run_emulator signature loss= replacing loss_mode=/berhu=;
  DELETE the run-level inertness guard and the phase_owns_berhu
  split; per-pass resolution; _PHASE_BLOCK_KEYS -> 7;
  validate_phase_block migration wording for the two retired keys;
  remove validate_berhu's None-mode hook), emulator/experiment.py
  (train() passes loss=; __init__ docstring),
  sweep_hyperparam_emulator_cosmic_shear.py (SWEEPABLE swap +
  comments), example_yamls x3 + tune comment (paste final blocks),
  notes/loss-mode-berhu.md (GB-C leg-2 recipe to the new schema).
- **Contracts & interfaces:** the schema above; loss()/_reduce
  UNCHANGED; absent block = {mode: sqrt} and a run with no loss
  block is byte-identical to today's default (GL-D golden); loud
  no-alias migration on BOTH paths. Declare any deviation.
- **Constraints & edge cases:** the berhu sub-block is valid only
  beside a berhu mode (local check — no cross-pass logic anywhere);
  phase loss blocks are full replacement (no partial mode-only
  overlay); knot tensors keep the kappa_t discipline.
- **Validation gate:** GL-A/B/C on the Mac (raw outputs); GL-D rides
  the workstation queue.
- **Next milestone:** IMPLEMENTER_HANDOFF with GL-A/B/C evidence +
  the updated GB-C recipe.

### 2026-07-06 — Architect re-audit: ACCEPTED (no deltas)

Verified independently (own harness, 27 checks, all PASS after one
reclassification). GL-A: all twelve validator cases — absent default,
mode-only berhu resolving default knots, per-mode berhu None for the
three plain modes, unknown mode naming the five, the LOCAL
berhu-sub-block-on-sqrt error, values via validate_berhu, TypeError,
and the dead None-mode hook confirmed removed (None now raises).
GL-B: the migration message carries mode AND knot/cap values in a
paste-ready nested block; flat keys inside phase blocks get the
migration wording through validate_phase_block; the top-level check
sits in resolve_phase_args BEFORE the two-phase early return — the
silently-dropped-flat-key trap is explicitly closed (the exact hole I
probed for). GL-C: whitelist 7 with "loss"; the D-B1 machinery is
GONE (zero grep hits for the guard and the split); run_emulator
signature carries loss= only; knot threading intact (kappa_t
discipline, both calls); SWEEPABLE swapped; no flat keys taught
anywhere in YAML/README; scans + whole-tree py_compile clean; the
GB-C leg-2 recipe reads the new schema. Reclassification, not a
deviation: loss_functions.py's diff includes error-message STRING
literals updated to the new key paths (train_args.loss.berhu.knot) —
strings are code to an AST but this is doc-coherence in substance,
and leaving the old paths would have been a bug.

The D-B1 arc closes fully: introduced by the flat schema, patched
same-day, then DELETED by the user's localization directive — the
structural fix beat the behavioral patch.

Commit (user):

    git add -A
    git commit -m "Nest loss options under train_args.loss (mode + berhu knots localized; D-B1 inheritance machinery deleted; early mode validation; gates GL-A-C Architect-verified)"
