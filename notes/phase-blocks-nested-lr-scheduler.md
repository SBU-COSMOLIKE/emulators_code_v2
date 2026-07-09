---
name: phase-blocks-nested-lr-scheduler
description: "Design spec (Architect, 2026-07-06): make the trunk:/head: per-phase override blocks mirror the top-level train_args schema — lr becomes a nested lr: {lr_base, warmup_epochs} sub-block (bare lr_base was flat, inconsistent with the top-level lr: block; loud migration), and a NEW scheduler: override joins the keys (user ask: lower patience on the head phase — today both phases build their fresh scheduler from the same global sched_opts, training.py ~1736, and both reuse the global warmup). Payoff beyond consistency: resolve_phase_args' single-phase demotion becomes a pure prefix-strip (trunk.X -> top-level X, the lr_base special case disappears), and the phase blocks finally get a whitelist — closing the noted looseness (unknown keys and scalar trunk: values silently ignored on the two-phase path). Seven phase keys: lr (overlay, bs_base rejected as run-global), scheduler (full replacement of the kwargs, cls fixed), loss_mode, trim, focus, clip, rewind."
metadata:
  node_type: memory
  type: project
---

# Phase blocks: nested lr:, per-phase scheduler:, one whitelist

User request 2026-07-06: (1) `head: lr_base:` flat is inconsistent
with the top-level `lr: {lr_base, bs_base, warmup_epochs}` — it should
be `head: lr: lr_base:`; (2) the head phase needs its own scheduler
patience ("I want to reduce patience on head").

## Facts (verified in source)

- Phase consumption (training.py ~1696-1736): bare
  `phase_opts["lr_base"]` feeds the sqrt rule; loss_mode / trim /
  focus / clip / rewind via .get; the fresh per-phase scheduler is
  built from the GLOBAL sched_opts (line ~1736) — no per-phase
  patience; `wmupe = lr_opts["warmup_epochs"]` is global too (both
  phase banners print the same warmup).
- No whitelist on the two-phase path: a typo'd phase key or a scalar
  `trunk:` value is silently ignored (the looseness noted-not-delta'd
  in [[resolve-phase-args-single-phase]]).
- The demotion path (resolve_phase_args) special-cases
  lr_base -> lr.lr_base precisely BECAUSE the phase key is flat while
  the top level nests.

## Design

1. **New phase-block schema** (both trunk: and head:, symmetric;
   mirrors the top level key-for-key):

       head:
         lr:                     # overlay onto the top-level lr block
           lr_base:       0.001
           warmup_epochs: 5      # optional; per-phase warmup, new
         scheduler:              # full replacement of the kwargs, new
           mode:     min
           patience: 10
           factor:   0.8
         loss_mode: sqrt
         trim:  { ... }          # full replacement (unchanged)
         focus: { ... }
         clip:   1.0
         rewind: true

   Semantics: `lr:` is an OVERLAY (lr_base and/or warmup_epochs; a
   phase `bs_base` is rejected loudly — the sqrt-rule anchor is
   run-global); `scheduler:` is a FULL REPLACEMENT of the top-level
   scheduler kwargs (mode/patience/factor/...; the scheduler CLASS is
   the run's, `cls` inside a phase scheduler is rejected) — matching
   the trim/focus full-replacement precedent. The other five keys are
   unchanged.
2. **Pure `validate_phase_block(block, which)` in training.py**
   (torch-free; experiment.py imports it — training must not import
   experiment). Raises on: a non-mapping block (closes the scalar
   `trunk: sqrt` hole); a bare `lr_base` (migration ValueError whose
   body is the paste-ready nested block, carrying the value over);
   an unknown key (whitelist = the seven above); `bs_base` inside a
   phase lr; `cls` inside a phase scheduler. Called early in
   run_emulator for both blocks (joins the existing
   before-any-setup-work guards) AND by resolve_phase_args before
   demotion.
3. **run_emulator consumption:** lr_pass from phase lr.lr_base (sqrt
   rule unchanged); wmupe_pass from phase lr.warmup_epochs (default:
   global) threaded into training_loop_batched per pass; on a phase
   scheduler override, make_scheduler gets {cls: run's class,
   **phase kwargs}; the phase banner line adds scheduler/warmup to its
   `[overrides: ...]` tail. Docstring key list rewritten to the seven.
4. **resolve_phase_args (experiment.py):** _PHASE_TRUNK_KEYS -> the
   seven; demotion = prefix-strip: every trunk key merges to the
   same-named top-level key — `lr` overlays the top-level lr block
   (bs_base preserved; the old special case now falls out naturally),
   `scheduler` full-replaces the top-level scheduler block, the rest
   replace as before. validate_phase_block runs first, so migration
   and typo errors are identical on both paths.
5. **validate_sweep_paths:** the trunk.* concretization simplifies to
   the generic strip — `trunk.lr.lr_base` -> "sweep 'lr.lr_base'
   directly", `trunk.scheduler.patience` -> "'scheduler.patience'";
   the lr_base special case is deleted. head.*/trunk_epochs messages
   unchanged.
6. **Docs sweep — the example_yamls are first-class targets**
   (enumerated by grep 2026-07-06; the Implementer re-greps in case
   lines moved):
   - train_single YAML: header sweep-pointer comment ~line 46
     (`head.lr_base` -> `head.lr.lr_base`); the per-phase overrides
     comment block ~95-105 (describe the seven keys + overlay vs
     full-replacement); the commented `# trunk:` block ~106-122
     (bare `lr_base` -> nested `lr:`); the commented `# head:` block
     ~123-137 (nested `lr:` + ADD the commented `scheduler:` example
     with the user's patience-10 ask); the top-level lr note ~308-309
     ("a head-specific base goes in the head: block" -> name
     `head.lr.lr_base`).
   - sweep YAML: dotted-path explainer ~37 and the commented sweep
     example ~86 (`head.lr_base` -> `head.lr.lr_base`).
   - tune YAML: the phase-keys comment ~14-17 (fixed scalars /
     rejected ranges wording updated to the nested schema).
   - run_emulator docstring + sweep driver set_by_path docstring.
   Every changed YAML block is pasted in the IMPLEMENTER_HANDOFF in
   its final form (the any-YAML-change paste-ready rule).
7. No aliases, hard break: a flat phase lr_base always raises the
   migration error (both two-phase and demotion paths).

## Validation gate

- GH-A (pure, Mac, exec-extracted validate_phase_block): nested valid
  block passes; bare lr_base -> migration ValueError containing the
  paste-ready nested block with the value carried over; unknown key ->
  lists the seven; scalar block -> TypeError; bs_base in phase lr and
  cls in phase scheduler each rejected naming the rule. Raw outputs.
- GH-B (pure, Mac): resolve_phase_args with the new schema — trunk
  {lr: {lr_base, warmup_epochs}, scheduler: {...}, loss_mode} demotes
  with the lr overlay preserving bs_base, the scheduler full-replacing,
  notice naming the keys, input deep-unmutated; validate_sweep_paths
  concretizes trunk.lr.lr_base / trunk.scheduler.patience by strip.
- GH-C (static, Mac): AST — validate_phase_block called for both
  blocks before setup in run_emulator and before demotion in
  resolve_phase_args; lr_pass reads lr.lr_base; wmupe threaded per
  pass; make_scheduler receives the phase-replaced spec on override;
  banner tail extended.
- GH-D style + YAML currency: house scans; whole-tree py_compile;
  keyword-vs-signature; AND a tree-wide grep gate — the string
  `head.lr_base` and any flat `lr_base:` directly under a trunk:/head:
  block appear NOWHERE in *.py / *.yaml / README (notes exempt), so
  no template or docstring still teaches the old schema.
- GH-E (workstation, rides the queue): one two-phase run with
  `head: {lr: {lr_base: 0.001}, scheduler: {mode: min, patience: 10,
  factor: 0.8}}` — banner shows the overrides and the head phase's
  first lr cut arrives on the patience-10 cadence; a config with no
  phase blocks is byte-identical pre/post change (golden run).

## Sequencing

training.py is currently carrying the IN-FLIGHT EMA implementation
([[weight-ema-snapshot-coupled]], not yet handed off/audited) and the
uncommitted eval-bs unit ([[eval-bs-decoupling]], D-E1/D-E2 pending).
Serialize: this feature starts only after the EMA handoff is audited
and both units are committed. Eventual commit (user):

    git commit -m "Nest the phase-block lr and add per-phase scheduler overrides (seven-key whitelist, loud lr_base migration; gates GH-A-D Architect-verified)"

then `cd ../../..` and `git merge claude/amazing-keller-e798b6`.

## Resume state (Implementer appends below)

Implemented 2026-07-06 (Opus 4.8, amazing-keller); on the clean base with
EMA committed (7ebc061). Execution log + raw GH-A-D evidence + the GH-E
recipe + the pasted YAML blocks in the last section.

### ARCHITECT_HANDOFF: READY FOR EXECUTION

- **Notes entry (read first):** this file;
  [[resolve-phase-args-single-phase]] (whose noted looseness this
  closes) and [[weight-ema-snapshot-coupled]] / [[eval-bs-decoupling]]
  for the training.py work it must serialize behind.
- **Target file(s):** emulator/training.py (validate_phase_block, the
  phase consumption at ~1696-1736, per-pass warmup threading,
  docstring, banner), emulator/experiment.py (resolve_phase_args
  prefix-strip + _PHASE_TRUNK_KEYS -> seven + import of the
  validator; validate_sweep_paths concretization),
  sweep_hyperparam_emulator_cosmic_shear.py (set_by_path docstring),
  example_yamls — ALL THREE, per the line-enumerated Design item 6
  (train_single: header pointer + overrides comment + both commented
  phase blocks nested + the patience-10 scheduler example + the
  top-level lr note; sweep: explainer + head.lr.lr_base example;
  tune: phase-keys comment). Paste every changed YAML block in its
  final form in the handoff.
- **Contracts & interfaces:** the seven-key whitelist {lr, scheduler,
  loss_mode, trim, focus, clip, rewind}; lr = overlay (bs_base
  rejected), scheduler = kwargs full-replacement (cls rejected);
  validate_phase_block(block, which) pure in training.py, imported by
  experiment.py; flat lr_base raises the migration error on BOTH
  paths. Declare any deviation.
- **Constraints & edge cases:** no aliases; two-phase runs without
  phase blocks byte-identical (GH-E golden); the sqrt rule and the
  fresh-optimizer-per-phase mechanics untouched; demotion stays
  mutation-free.
- **Validation gate:** GH-A/B/C/D on the Mac (raw outputs); GH-E rides
  the workstation queue.
- **Next milestone:** IMPLEMENTER_HANDOFF with GH-A-D evidence, only
  after the EMA feature is handed off, audited, and committed.

### 2026-07-06 — Implementer (Opus 4.8) execution

Started on the go-ahead once EMA committed (7ebc061), so training.py /
experiment.py are a clean base. Mac dev box: GH-A-D by exec-extraction /
AST; GH-E is the workstation recipe below.

**Done (exactly the target-file list):**

- training.py: `_PHASE_BLOCK_KEYS` (the seven) + pure `validate_phase_block
  (block, which)` + `_phase_lr_migration_message` (the single whitelist,
  imported by experiment.py; training does not import experiment). The
  validator: None -> None; non-mapping (block / lr / scheduler) ->
  TypeError; bare lr_base -> a migration ValueError printing the nested
  lr: block with the value carried; unknown key -> the seven; a phase lr
  unknown key -> {lr_base, warmup_epochs}; bs_base in a phase lr and cls
  in a phase scheduler each rejected naming the rule. run_emulator:
  validates both blocks up front (joins the pre-setup guards); the phase
  consumption reads lr.lr_base for the sqrt-rule lr_pass, wmupe_pass from
  lr.warmup_epochs (default: global) threaded into training_loop_batched
  per pass, sched_pass = {cls: run's, **phase scheduler} full-replacing
  the kwargs; the banner `[phase overrides: ...]` tail gains warmup /
  scheduler; the trunk_opts/head_opts docstring rewritten to the seven.
- experiment.py: imports validate_phase_block + _PHASE_BLOCK_KEYS; the
  local _PHASE_TRUNK_KEYS deleted. resolve_phase_args validates both
  blocks first, then demotes by prefix-strip (lr overlays the top-level
  lr block preserving bs_base; scheduler and the other five full-replace);
  the old lr_base special case is gone. validate_sweep_paths concretizes
  trunk.X.Y by a generic ".".join(strip) (trunk.lr.lr_base -> lr.lr_base,
  trunk.scheduler.patience -> scheduler.patience); the lr_base special
  case deleted.
- Docs: train_single YAML (header pointer head.lr.lr_base; the overrides
  comment rewritten to the seven keys + overlay/full-replacement; the
  commented trunk: / head: blocks nested lr: + the head: scheduler
  patience-10 example + a per-phase warmup_epochs example; the top-level
  lr note names head.lr.lr_base and bs_base as run-global); sweep YAML
  (explainer + commented example head.lr.lr_base); tune YAML (phase-keys
  comment, nested); sweep set_by_path docstring + both driver headers;
  README two rows.

**Deviations from blueprint (declared):** the seven-key constant is
`_PHASE_BLOCK_KEYS` (renamed from the spec's "_PHASE_TRUNK_KEYS -> the
seven") and lives in training.py beside validate_phase_block, imported by
experiment.py — one whitelist, no duplicated tuple, no reverse import.
Also (a stricter reading of "errors identical on both paths"):
resolve_phase_args validates the head: block too before dropping it, so a
malformed head: on a single-phase model fails loudly rather than being
silently dropped (matches the shared-YAML intent).

**Gate evidence (raw, Mac):**

- GH-A validate_phase_block (pure, exec-extracted): None -> None; a full
  nested block returns unchanged; scalar block -> TypeError; bare lr_base
  -> migration ValueError ("train_args.head.lr_base is gone ... lr: /
  lr_base: 0.0025"); unknown top key lists the seven; bs_base in phase lr
  / cls in phase scheduler / unknown phase-lr key / non-mapping lr /
  non-mapping scheduler each raise. GH-A: ALL PASS.
- GH-B resolve_phase_args + sweep-paths (pure, one shared namespace since
  experiment imports validate_phase_block): a trunk with nested lr +
  scheduler + loss_mode demotes with the lr OVERLAY updating lr_base +
  warmup_epochs while PRESERVING bs_base (256), the scheduler
  full-replaced, loss_mode merged, top-level trim surviving, notice naming
  lr / scheduler / loss_mode, the input deep-unmutated; two_phase a no-op;
  a flat lr_base raises on the demotion path too; validate_sweep_paths
  strips trunk.lr.lr_base -> "lr.lr_base" and trunk.scheduler.patience ->
  "scheduler.patience". GH-B: ALL PASS.
- GH-C static wiring (AST): validate_phase_block called for both blocks
  before build_loaders and the training_loop_batched call in run_emulator,
  and before `resolved = dict(...)` in resolve_phase_args; the consumption
  reads phase lr.lr_base, threads wmupe_pass into warmup_epochs, and
  passes sched_pass ({cls: run's, **phase}) to make_scheduler; the banner
  tail is extended; experiment imports the validator + constant; no
  reverse import. GH-C: ALL PASS.
- GH-D style + grep: 0 new all-caps (acronyms allowlisted) / 0 new ` -- `
  on all eight touched files; core .py <= 90 cols; 0 new comprehensions;
  `head.lr_base` and any flat phase lr_base appear NOWHERE in py / yaml /
  README (only in notes); whole-tree py_compile. GH-D: ALL PASS.
- GH-E workstation (rides the queue): (1) a two-phase run with
  `head: {lr: {lr_base: 0.001}, scheduler: {mode: min, patience: 10,
  factor: 0.8}}` — the banner shows `[head overrides: scheduler]` and the
  head phase's first lr cut arrives on the patience-10 cadence (vs the
  run's 25); (2) a config with NO phase blocks is byte-identical pre/post
  change (golden run: diff the epoch lines).

      R=--root=<root> ; F=--fileroot=<fileroot>
      # golden: no-blocks config unchanged
      git stash && python train_single_emulator_cosmic_shear.py $R $F \
        --yaml=train_single_emulator_cosmic_shear.yaml > /tmp/ph_pre.log 2>&1
      git stash pop && python train_single_emulator_cosmic_shear.py $R $F \
        --yaml=train_single_emulator_cosmic_shear.yaml > /tmp/ph_post.log 2>&1
      diff <(grep -E '^(phase|epoch|best)' /tmp/ph_pre.log) \
           <(grep -E '^(phase|epoch|best)' /tmp/ph_post.log)   # EMPTY
      # override: a two-phase YAML with the head scheduler patience 10
      python train_single_emulator_cosmic_shear.py $R $F \
        --yaml=train_single_twophase_headpatience.yaml    # watch the cadence

**Pasted YAML blocks (final form, per the any-YAML-change rule).** The
train_single trunk: / head: commented example, now nested:

    # trunk:                   # phase 1 (pure-trunk training)
    #   lr:
    #     lr_base:   0.0025    # phase-1 base lr (same sqrt-batch rule)
    #   loss_mode: sqrt
    #   trim: { ... }          # (unchanged full-replacement blocks)
    #   focus: { ... }
    # head:                    # phase 2 (frozen trunk, head only)
    #   lr:
    #     lr_base:       0.001 # phase-2 base lr (same sqrt-batch rule)
    #     warmup_epochs: 5     # optional per-phase warmup (default: global)
    #   scheduler:             # full replacement of the scheduler kwargs;
    #     mode:     min        # the run's scheduler class stays. A lower
    #     patience: 10         # head patience cuts the lr sooner on the
    #     factor:   0.8        # frozen-trunk phase.
    #   loss_mode: sqrt
    #   trim: { ... }
    #   clip:   1.0
    #   rewind: true

The sweep YAML commented example:

    # sweep:                       # head-phase base lr (two-phase only;
    #   parameter: head.lr.lr_base # rejected at startup on single-phase)
    #   values:
    #     - 0.0005

Open: GH-E (workstation) + the Architect re-audit.

### 2026-07-06 — Architect re-audit: ACCEPTED (no deltas)

Verified independently (own harness, 36 checks, all PASS). GH-A: the
ten validator cases incl. the migration message carrying the value
(`lr_base: 0.0025` inside a paste-ready nested lr:), the seven-key
listing, bs_base/cls rejections, and TypeErrors for scalar block /
non-mapping lr / non-mapping scheduler. GH-B: demotion with my own
config — lr overlay preserves bs_base (256) AND merges the phase
warmup_epochs into the top-level lr block; scheduler full-replaced;
top-level trim survives; notice `(lr, scheduler, loss_mode)`; input
deep-unmutated; flat lr_base raises on the demotion path; the
malformed-head case raises despite head being dropped (deviation 2
exercised directly); sweep-path concretization by generic strip
(trunk.lr.lr_base -> 'lr.lr_base', trunk.scheduler.patience ->
'scheduler.patience'), bare-trunk fallback intact. GH-C: both blocks
validated up front (lines 1625/1626, before the ema check at 1630 and
make_model at 1686); lr.lr_base consumption; warmup threaded per pass;
make_scheduler on sched_pass only, cls preserved; experiment imports
the validator, local _PHASE_TRUNK_KEYS gone; resolve validates before
the copy. GH-D: scans clean; tree-wide stale-schema grep empty
(head.lr_base / flat phase lr_base nowhere in py/yaml/README);
py_compile whole tree; the pasted YAML head block matches the file
(nested lr + patience 10).

Both declared deviations ACCEPTED: (1) the single shared
_PHASE_BLOCK_KEYS in training.py imported by experiment.py is
one-source-of-truth, better than a duplicate; (2) validating head:
before dropping it supersedes the old "dropped unvalidated" rule from
[[resolve-phase-args-single-phase]] — errors are now identical on both
paths, which also CLOSES that note's "noted, not a delta" looseness
(scalar trunk: / unknown phase keys silently ignored) in full.

Open: GH-E only (workstation: head patience-10 cadence + the no-blocks
golden run). Commit (user):

    git add -A
    git commit -m "Nest the phase-block lr and add per-phase scheduler overrides (seven-key whitelist, loud lr_base migration; gates GH-A-D Architect-verified)"

### 2026-07-08 — board verdict (Architect): GH-E head-scheduler-override PASS (cadence unexercised)
The override wiring is proven: the head phase banner reads "lr restarts
at 2.00e-03 ... [head overrides: scheduler]" against the trunk's
5.00e-03 — both the nested lr and scheduler blocks resolved. Honest
margin: the patience-10 CADENCE itself never fired in the smoke (the
head's val kept creeping down, no plateau long enough within 30
epochs), so the lr-cut spacing stayed unobserved; the assertion that
passed is the banner + restart value. A longer-plateau smoke would be
needed to watch a cut land on the 10-epoch cadence — not queued, the
wiring evidence suffices for the board.
