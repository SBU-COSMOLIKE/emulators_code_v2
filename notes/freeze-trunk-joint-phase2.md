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
