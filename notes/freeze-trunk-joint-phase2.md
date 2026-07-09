---
name: freeze-trunk-joint-phase2
description: "SPEC 2026-07-06 (Architect): train_args.freeze_trunk (bool, default true) — phase 2 of the two-phase schedule may train trunk AND head together (joint fine-tune) instead of freezing the trunk. User directive: 'a key allowing trunk and head to trained together'; name recommendation freeze_trunk: false (veto-able; positive alternative joint_finetune: true). Tiny feature: set_train_phase ALREADY accepts 'joint' (everything trains, head active) at IA/emulator_designs.py:531/845 and the per-pass optimizer covers every param with requires_grad doing the freezing — run_emulator just never calls it; the change is phase-2 phase string = 'head' if freeze_trunk else 'joint', validation (needs trunk_epochs > 0; bool; demotion drops it like trunk_epochs; sweep-guard on single-phase), the banner fragment '(two-phase: N trunk + M joint)', and the LICENSE RULE (user directive): a per-head activation (canonical pin or head: alias) requires a frozen-trunk head phase — trunk_epochs > 0 AND freeze_trunk true — else a teaching error; extension to the trunk_epochs-0 case included, veto-able. Lands as COMMIT UNIT A of the head-activation handoff cycle (the license check in unit B reads this key). Gates GFT-A..C. NOT implemented."
metadata:
  node_type: memory
  type: project
---

# train_args.freeze_trunk: a joint phase 2 (spec)

User directive 2026-07-06: "We should have a key allowing trunk and
head to trained together (right now trunk is trained, freeze, then
head is trained). If that key is on, then having the activation
inside the head model should throw an error."

## What exists already (verified at 8bb5484)

- Plain joint training from epoch 1 exists TODAY: trunk_epochs 0 (or
  absent) on a Template model trains everything together (no phase
  calls; the constructor default phase is joint).
- The MODEL capability for a joint phase exists TODAY:
  set_train_phase (IA/emulator_designs.py:531 TemplateResCNN, :845
  TemplateResTRF) accepts "joint" | "trunk" | "head"; "joint" =
  everything trains, head active. run_emulator simply never calls
  "joint" — its two-phase loop is hardwired "trunk" then "head".
- The per-pass optimizer is built over every parameter and the
  freeze acts through requires_grad (training.py ~2258), so a joint
  phase 2 needs NO optimizer change: flipping the phase string is
  sufficient mechanically.

So the new mode is: phase 1 trains the trunk alone (unchanged, at
trunk cost), then phase 2 trains trunk AND head together (a joint
fine-tune warm-started by phase 1) instead of freezing the trunk.
Loss continuity at the handoff is untouched (the head starts at its
zero-init identity either way). Phase-2 epochs cost MORE than the
frozen mode (the trunk backward returns) — a documented fact, and
the GFT-C sanity signal.

## The key

    train_args.freeze_trunk: true | false     (top level, beside
                                               trunk_epochs; default
                                               true = today, absent =
                                               byte-identical)

    # freeze_trunk: false   # phase 2 trains trunk + head together
                            # (joint fine-tune warm-started by phase
                            # 1) instead of freezing the trunk at the
                            # handoff. Needs trunk_epochs > 0.

Name RECOMMENDATION (veto-able): freeze_trunk, default true — the
key names the exact schedule fact that changes, and its default
states today's behavior; matches the house boolean style
(shared_mlp, rescale_kernel, rewind). Positive-polarity alternative
if the user prefers "the key is ON for the new mode":
joint_finetune: true. (The model API's own name for the mode is
"joint".)

## Semantics and validation

- freeze_trunk present (either value) without trunk_epochs > 0 ->
  ERROR (the trunk:/head: "would silently do nothing" pattern).
- Two-phase run: phase 2 calls set_train_phase("head" if
  freeze_trunk else "joint"). Phase 1 unchanged. The head: override
  block still configures phase 2 whatever the mode (docs sentence:
  head: is the PHASE-2 block; with freeze_trunk false its lr /
  scheduler / loss / trim / focus / clip / rewind / ema govern the
  joint pass).
- Single-phase model (no set_train_phase): resolve_phase_args drops
  freeze_trunk with the notice, exactly like trunk_epochs;
  validate_sweep_paths rejects it as a sweep/search axis on
  single-phase, exactly like trunk_epochs.
- Sweepable / searchable on two-phase models as an ordinary boolean
  leaf (false/true — the film sweep pattern).
- Banner (consumed view, the directive): the run line's two-phase
  fragment becomes "(two-phase: N trunk + M joint)" when false
  ("head" unchanged when true/absent — pending gate recipes that
  grep the default text stay valid); the phase-2 banner line names
  the phase accordingly.

## The license rule (the user's directive, one rule)

A per-head activation — the canonical model.cnn/.trf.activation pin
OR the head: activation: alias — is licensed by the head training
ALONE in phase 2. Requirement: trunk_epochs > 0 AND freeze_trunk
true (on the active head). Violations ERROR at build_specs (the
consumption site) with a teaching message:

    model.trf.activation: a per-head activation needs a
    frozen-trunk head phase (trunk_epochs > 0 and freeze_trunk
    true): the head family IS the head-phase family. With the
    trunk and head training together the network keeps one family
    — set model.activation only.

This covers BOTH together-regimes: freeze_trunk false (the user's
exact ask) AND trunk_epochs 0/absent on a Template model (same
physics: the head never trains alone; Architect extension,
veto-able). An inactive head block on resmlp stays ignored as
today. The check lives in the head-activation feature (commit unit
B), which reads this key.

## Gates

- GFT-A (Mac, pure/exec-extract): validation — non-bool rejected;
  either value without trunk_epochs > 0 errors; absent = default
  true path byte-identical; demotion drops it with the notice;
  validate_sweep_paths rejects freeze_trunk on single-phase, allows
  it on two-phase.
- GFT-B (Mac, static): the phase-2 call site is exactly
  '"head" if ... else "joint"'; no optimizer-path change (the
  per-pass build over every param is untouched — verify the ~2258
  claim in the tree and cite the line); banner fragment gated on
  the key with the default text byte-identical; the license check
  sits in build_specs beside the pin translation.
- GFT-C (workstation, rides the queue): restrf + nla, small
  trunk_epochs, freeze_trunk false — phase-2 banner says joint,
  loss continuous at the handoff, phase-2 epoch time visibly above
  the frozen-mode control (the trunk backward is back); golden
  absent-key byte-identity run.

## Packaging

COMMIT UNIT A of the head-activation handoff cycle
([[head-activation-per-component]]): A = this key (training
schedule), B = per-head activation (its license check reads A),
C = the README precedence appendix ([[readme-precedence-appendix]],
which gains the freeze_trunk rows). One Implementer session, three
commit units, after the doc-audit commit lands.

## Status

SPEC DELIVERED 2026-07-06, NOT implemented. Suggested commit
sentence (unit A): "train_args.freeze_trunk (default true): phase 2
may train trunk + head together (set_train_phase('joint') existed,
now reachable; needs trunk_epochs > 0; demotion + sweep guards
mirror trunk_epochs; banner says joint; gates GFT-A/B
Architect-verified)".

## Resume state (Implementer appends below)

### 2026-07-06 — Implementer (Opus 4.8): UNIT A executed

Clean base: the doc-audit commit landed at 2d2f68d; started there.
GFT-A/B pass on the Mac. This is commit unit A of the three-unit
head-activation cycle; B (per-head activation) and C (precedence
appendix) follow after A is committed (shared files -> sequential
commits; B's license check reads freeze_trunk).

**Done:**

- training.py: run_emulator gains `freeze_trunk=None` (None = absent =
  the frozen default, byte-identical). Early guard beside the trunk_opts
  guard: non-bool -> TypeError; explicit value with trunk_epochs == 0 ->
  ValueError ("would silently do nothing"). Phase logic:
  `freeze = freeze_trunk is not False`; the pass ROLE stays "trunk" /
  "head" (selects opts, drives the best-epoch restore, labels the tail),
  and only the model phase name changes:
  `model_phase = "head" if freeze else "joint"` for the head pass,
  consumed by `set_train_phase(model_phase)` (single-phase phase=None ->
  model_phase None -> no call). The per-pass banner prints model_phase
  ("head" frozen = byte-identical, "joint" when false). The optimizer
  build (make_optimizer over every param; freeze via requires_grad) is
  UNCHANGED. Docstring: the two-phase diagram + the freeze_trunk arg
  entry; D-DOC2b obs2 (the diagram legend gains scheduler + ema).
- experiment.py: train() passes `freeze_trunk=train_args.get("freeze_trunk")`;
  resolve_phase_args demotes freeze_trunk on a single-phase model (drops
  it, names it in the notice, like trunk_epochs) and lists it in
  has_phase; validate_sweep_paths rejects a freeze_trunk sweep axis on
  single-phase; print_design's two-phase fragment reads
  "(two-phase: N trunk + M joint)" when freeze_trunk false ("head"
  byte-identical otherwise); __init__ docstring gains the freeze_trunk
  entry + the demotion sentence.
- sweep_hyperparam: "freeze_trunk" joins SWEEPABLE_TOP_KEYS.
- train_single YAML: the freeze_trunk explanation + commented key beside
  trunk_epochs; the phase-2 prose now says "by default"; D-DOC2b obs3
  (the phase-block full-replacement walkthrough gains ema).
- DEFERRED to unit B (per the re-audit's D-DOC2b assignment + ruling d):
  the license check (a per-head activation needs trunk_epochs > 0 AND
  freeze_trunk true) lives in build_specs, which unit B adds; obs1 (the
  experiment.py train() `(lr / loss / trim / focus)` enumeration) is
  unit B's. Unit A leaves both untouched.

**Deviations from spec:** none behavioral. Declared interface addition:
run_emulator gained a `freeze_trunk` kwarg (default None); experiment.train
passes it. The phase-2 call uses a `model_phase` local (the role/name
split) so head_opts, banner-role, and the best-epoch restore keep keying
off the "head" role while set_train_phase receives "joint" — the note's
"'head' if ... else 'joint'" literal is exact at the call site.

**Gate evidence (raw, Mac; no torch — exec-extracted from the tree):**

- GFT-A (validation / demotion / sweep guard): resolve_phase_args
  two-phase no-op (freeze_trunk preserved) vs single-phase drop
  (notice "single-phase model: trunk_epochs and freeze_trunk ignored";
  freeze_trunk-only also demotes); validate_sweep_paths allows the axis
  on two-phase, rejects it loudly on single-phase; the run_emulator guard
  over 9 cases (None any trunk_epochs -> no raise; True/False +
  trunk_epochs 0 -> ValueError; str/int/float -> TypeError; valid
  two-phase -> no raise); freeze = `is not False` (None/True->True,
  False->False) and model_phase (trunk->trunk, head+freeze->head
  byte-identical, head+joint->joint, None->None). ALL PASS.
- GFT-B (static): the call-site literal `"head" if freeze else "joint"`
  present at training.py; the make_optimizer call text byte-identical to
  HEAD (optimizer path untouched — the ~2258 "collects every parameter"
  comment intact); print_design phase2 gating (absent/true -> "head"
  byte-identical, false -> "joint"); whole-tree py_compile clean; house
  scans clean (0 over 90 cols, 0 double-dash, 0 new caps emphasis after
  de-capping one "ROLE" comment). AST-minus-docstrings: training.py,
  experiment.py, sweep driver all CODE CHANGED as expected.

**GFT-C (workstation, rides the queue).** Run from $ROOTDIR on a torch
box, two-phase model (restrf + ia nla), small trunk_epochs:

    R=--root=<root> ; F=--fileroot=<fileroot>
    Y=--yaml=train_single_emulator_cosmic_shear.yaml
    # golden: absent freeze_trunk is byte-identical pre/post the feature
    git stash && python train_single_emulator_cosmic_shear.py $R $F $Y \
      > /tmp/ft_pre.log 2>&1
    git stash pop && python train_single_emulator_cosmic_shear.py $R $F $Y \
      > /tmp/ft_post.log 2>&1
    diff <(grep -E '^(phase|epoch|best|run:)' /tmp/ft_pre.log) \
         <(grep -E '^(phase|epoch|best|run:)' /tmp/ft_post.log)   # EMPTY
    # then a joint run: set (restrf/nla) trunk_epochs: 800, freeze_trunk: false
    #   startup: "run: ... (two-phase: 800 trunk + M joint)"
    #   phase 2: "phase 'joint': M epochs, lr restarts ..."
    #   loss continuous at the handoff (the zero-init head starts identity);
    #   phase-2 epoch time visibly ABOVE a freeze_trunk: true control run
    #   (the trunk backward returns) — the GFT-C sanity signal.

Open: GFT-C (workstation) + the Architect re-audit of GFT-A/B. Unit A
commit is independent; units B/C follow on the committed base.

### 2026-07-06 — Architect re-audit: ACCEPTED, no deltas

Own harnesses, base 2d2f68d confirmed:

- Footprint exact: AST-minus-docstrings — CODE CHANGED only in
  training.py / experiment.py / sweep_hyperparam driver; YAML
  doc-only; all other files CODE-IDENTICAL.
- GFT-A reproduced 14/14 (exec of the real resolve_phase_args /
  validate_sweep_paths; the run_emulator guard exec-extracted
  verbatim via ast): two-phase no-op preserves the key;
  single-phase drops it with the notice naming it (freeze_trunk-only
  YAML also demotes); input unmutated; sweep axis rejected
  single-phase / allowed two-phase; guard passes None/True/False
  legal cases, raises ValueError on explicit + trunk_epochs 0 and
  TypeError on "false" / 1 / 0.0 (int/float correctly non-bool);
  the freeze / model_phase truth table (9 cells) exact from the
  verbatim expressions.
- GFT-B reproduced: the '"head" if freeze else "joint"' literal at
  the call site; zero make_optimizer hunks (optimizer path
  untouched); print_design fragment defaults to "head"
  (byte-identical text, tk-gated); banner prints model_phase (trunk
  and frozen-head labels unchanged); D-DOC2b obs2 (diagram legend
  now eight keys) + obs3 (YAML walkthrough gains ema) both paid;
  scans clean (0 long lines, 0 double-hyphen, caps = YAML only);
  py_compile OK.
- Declared deviation ACCEPTED: the role/name split (pass role stays
  "trunk"/"head" for opts / restore / tail; only the
  set_train_phase name flips to "joint") is the right refinement —
  head_opts semantics ("head: is the phase-2 block in either mode")
  hold exactly as the spec's docs sentence requires. run_emulator's
  freeze_trunk=None kwarg + experiment.train pass-through accepted
  as the declared interface addition.
- Noted, not a delta: a non-bool freeze_trunk renders as "head" in
  print_design before run_emulator's TypeError stops the run — the
  same banner-before-guard ordering every run_emulator-guarded key
  has; the run never executes against the wrong display.

UNIT A COMMIT-READY (sentence above). Units B/C proceed on the
committed base; GFT-C rides the workstation queue.

### 2026-07-08 — board verdict (Architect): GFT-C joint-training PASS
Banners exact ("two-phase: 30 trunk + 10 joint", phase 'joint'); the
freeze_trunk:true control runs the same physics as today's head phase.
The epoch-time sanity signal holds with the expected small margin on
this GPU: joint steady 0.5 s/epoch ABOVE the control's 0.4 (the trunk
backward returned). Loss continuous at the handoff (joint epoch 1 val
5.92 from the 12.73 baseline, no restart spike). Green runs 3-11.
