---
name: readme-precedence-appendix
description: "SPEC 2026-07-06 (Architect): a new README appendix 'Precedence: who wins when settings collide' — user directive after liking the activation-precedence table ('we need there an appendix showing many examples of precedence in different flags'). Content inventory (Architect-verified in the tree, two rows flagged verify-at-build): the activation slots table (flag > model.activation > H for the shared slot; per-head pin holds + warning — needs the head-activation feature, so this appendix LANDS WITH IT as commit unit 2 of that handoff); phase-block-vs-top-level per-key semantics (overlay / full-replace / value / ema null; bs_base + scheduler cls forbidden); single-phase demotion (trunk > top level, head/trunk_epochs dropped); loss-block spellings (berhu: vs mode-named, both = error, no silent winner); sweep/tune precedence; n_train / thresholds / ram_frac / compile_mode constructor-vs-YAML rows; and the deliberately-no-knob table (derived eval bs, run-global bs_base, fixed optimizer/scheduler classes, TRF token width + MLP interior width pin D-DOC9)."
metadata:
  node_type: memory
  type: project
---

# README appendix: precedence — who wins when settings collide (spec)

User directive 2026-07-06 (verbatim intent): the activation
precedence table belongs in the README, inside a NEW appendix
"showing many examples of precedence in different flags", because
activation is not the only setting with preference rules.

Packaging: the flagship table includes model.trf.activation, which
exists only after the head-activation feature — so this appendix
LANDS WITH that feature, as COMMIT UNIT 2 of the head-activation
handoff ([[head-activation-per-component]]). Placement: a new
numbered appendix in the README contents (after the activation
appendix, before "every file's functions"); the Implementer renumbers
the anchors.

Every rule below was verified in the tree by the Architect except
the two marked (verify at build). One table per subsection; each
table gets a one-line "why" underneath, matching README house style.

## A. The activation slots (the flagship table)

Two slots: the SHARED slot (trunk + every component without its own
key) and the HEAD slot. Shared slot: --activation flag >
model.activation > "H". Head slot: model.cnn/.trf.activation pins it;
otherwise it follows the shared slot.

| --activation | model.activation | model.trf.activation | trunk gets | head gets |
|---|---|---|---|---|
| absent | H | absent | H | H (shared) |
| absent | H | gated_power | H | gated_power (pin) |
| power | H | absent | power | power (flag wins the shared slot) |
| power | H | gated_power | power | gated_power (pin holds; a startup WARNING prints) |

Footnotes the appendix must carry:
- The flag sets only the TYPE; n_gates is always read from the YAML
  (model.activation.n_gates for the shared slot, the per-head
  block's own n_gates for a pin) — a subtle rule, deserves its row.
- The bake-off and a model.activation sweep write the shared slot
  per curve: a pinned head stays pinned across all curves (warning
  at startup).
- The alias: on a two-phase model, head: activation: is accepted as
  a second spelling of the pin (canonical =
  model.cnn/.trf.activation, which also works on single-phase
  YAMLs); BOTH spellings given = error, keep one, even when equal.
  trunk: activation: is an ERROR — the asymmetry is deliberate (the
  trunk is the same modules in both phases; see section B's row).
- The license: a per-head activation (either spelling) requires a
  frozen-trunk head phase — trunk_epochs > 0 AND freeze_trunk true.
  With the trunk and head training together (freeze_trunk: false,
  or no two-phase schedule at all) the network keeps one family;
  the pin errors with a teaching message.

## B. Phase blocks vs the top level (two-phase models)

Per key, phase-block value > top-level train_args > built-in
default — but the OVERRIDE SEMANTICS differ by key, and that is the
table:

| key | semantics | notes |
|---|---|---|
| lr | OVERLAY {lr_base, warmup_epochs} onto the top-level lr | bs_base is run-global: inside a phase lr it is an ERROR |
| scheduler | full kwargs replacement | the scheduler CLASS stays the run's: cls in a phase is an ERROR |
| loss | full block replacement | {mode, berhu}; no merging with the top-level block |
| trim / focus | full block replacement | restart at the phase's own epoch 1 |
| clip / rewind | value replacement | |
| ema | full block replacement | ema: null = explicit per-phase OFF, overriding an inherited top-level ema |
| activation | HEAD ONLY: alias for model.<head>.activation (not a training knob; consumed at construction) | LEGAL in head: (the head only trains in phase 2); ERROR in trunk: — the trunk is the same modules in both phases, so a phase-local trunk activation cannot exist (the error message explains this) |

Why: phase blocks are DIFFS against the top level, not containers
(the bs_base ruling); construction knobs are run-global — with ONE
deliberate, user-ruled exception: head: activation:, coherent only
because the head component and the head phase coincide.

## C. Single-phase demotion (the same YAML on resmlp)

On a model without set_train_phase: trunk: keys WIN over top-level
keys (trunk merges into the top level, full-replace per key; lr
overlays preserving bs_base); head:, trunk_epochs, and freeze_trunk
are DROPPED; a one-line notice prints. The user's rule: "what is in
the trunk is just the global."

## C2. The two-phase schedule modes

| trunk_epochs | freeze_trunk | schedule |
|---|---|---|
| 0 / absent | (must be absent) | joint training from epoch 1 |
| N > 0 | true / absent | trunk-only for N epochs, then FROZEN trunk + head-only (today's default) |
| N > 0 | false | trunk-only for N epochs, then trunk + head TOGETHER (joint fine-tune) |

freeze_trunk without trunk_epochs > 0 is an error (it would
silently do nothing). The head: override block configures phase 2
in either mode.

## D. The loss block spellings

- loss absent, or no mode key -> mode sqrt.
- A berhu mode without a berhu: sub-block -> knot 0.2, cap 10.
- The knot sub-block may be spelled berhu: (family, sweep-safe) or
  exactly after the active mode (berhu_capped: under mode
  berhu_capped). BOTH present -> ERROR: no silent winner, ever.

## E. Sweeps and searches

| context | rule |
|---|---|
| sweep_hyperparam | the sweep leaf > the train_args baseline (deep copy per point) |
| sweep parameter model.activation | special case: sets the experiment's shared slot; the --activation flag must be unset |
| model.name / model.ia | REFUSED as sweep axes (they change the class) |
| phase axes on a single-phase model | REFUSED at startup (validate_sweep_paths) |
| tune ranges [default, min, max, kind] | the suggested value per trial > the default; the default is what the train drivers use and what warm-starts trial 0 |
| sweep_ntrain | the driver's per-point n_train > data.n_train (stage_train argument) |

## F. Constructor / driver args vs YAML

| setting | precedence |
|---|---|
| device | explicit arg > auto-detect (CUDA > MPS > CPU) |
| thresholds | constructor arg > DEFAULT_THRESHOLDS |
| rescale | driver flag only (no YAML key) |
| ram_frac | the parallel sweep workers force 0 (stream from the shared memmap) > data.ram_frac (verify at build: the exact mechanism) |
| compile_mode | model.compile_mode (YAML) > the architecture default (conv / TRF: "default"; resmlp: reduce-overhead) (verify at build: the resmlp default) |

## G. Deliberately NO knob (anti-precedence: nothing to win)

| setting | why there is no override |
|---|---|
| eval batch size | derived from n_val (~1024 target, minimal tail pad); decoupled from the training bs by design |
| bs_base | run-global sqrt-rule anchor; a phase-level value is an error |
| optimizer / scheduler class | fixed (AdamW / ReduceLROnPlateau) in the drivers; a phase scheduler overrides kwargs only |
| TRF token width | pinned to the padded bin length (physics; no embedding, no adapters) |
| TRF MLP interior width | pinned to the token width (user ruling, [[trf-mlp-width-knob]] shelved; n_mlp_blocks is depth only) |

## Gate

GPR-A (rides the head-activation cycle): every table row
cross-checked against the tree at build time (the Implementer
verifies the two flagged rows and corrects them if the code says
otherwise, declaring the correction); anchors + contents renumbered;
house scans on the diff.

## Status

SPEC DELIVERED 2026-07-06, NOT implemented. Lands as COMMIT UNIT 2
of the head-activation handoff (the flagship table needs the
feature). The head-activation note's handoff references this note.
