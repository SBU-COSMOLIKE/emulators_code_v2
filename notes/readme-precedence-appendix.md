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

## Proceed directive (Architect, 2026-07-06, after unit B acceptance)

Units A (freeze_trunk, ebd9869) and B (per-head activation,
Architect-ACCEPTED 30/30, re-audit verdict in
[[head-activation-per-component]]) are done. Unit C is AUTHORIZED on
B's committed base:

- Precondition: `git log -1` shows the unit-B sentence
  ("Per-head-component activation: ..."). Uncommitted unit-B changes
  in the tree = STOP and report.
- Scope: the appendix exactly per this note — sections A..G
  (including C2, the schedule-modes table) with the flagship
  activation table, the alias / license / n_gates footnotes, the
  freeze_trunk rows, and the deliberately-no-knob table. Verify the
  two verify-at-build rows (the parallel-worker ram_frac mechanism;
  the resmlp compile_mode default) against the tree and CORRECT them
  if the code disagrees, declaring the correction. New numbered
  README appendix (after the activation appendix, before "every
  file's functions"); renumber the contents + anchors. House style:
  one table per subsection, a one-line "why" beneath each.
- This unit is DOCUMENTATION-ONLY: every .py and .yaml file stays
  byte-identical. Finding a reason to touch code = STOP and report.
- Gate GPR-A: every table row cross-checked against the tree;
  anchors resolve; house scans on the diff (README one-line rows
  exempt from the 90-col rule per file convention); py_compile moot
  but run it anyway on the tree.
- Report: IMPLEMENTER_HANDOFF + resume state appended to THIS note,
  raw gate evidence, deviations declared. Do not commit: print the
  suggested commit command.

## Resume state (Implementer appends below)

### 2026-07-06 — Implementer (Opus 4.8): UNIT C executed

Base: unit B committed at a95f8df; started there. Documentation-only —
git diff touches ONLY README.md (+ this note); every .py and .yaml is
byte-identical (GPR-A confirmed). Commit unit C, the last of the cycle.

**Done:**

- New README appendix "## 10. Appendix: precedence — who wins when
  settings collide", inserted after the activation-functions appendix
  (section 9) and before "every file's functions"; the latter renumbered
  to "## 11". Contents gained the section-10 line and the section-11
  renumber. Sections A (activation slots + the 4 footnotes), B (phase-
  block per-key semantics), C (single-phase demotion), C2 (schedule
  modes), D (loss spellings), E (sweeps/tune), F (constructor/driver vs
  YAML), G (deliberately-no-knob) — one table per subsection, a one-line
  "Why:" beneath each, house style.
- Anchors: the two new/renumbered auto-anchors resolve
  (#10-appendix-precedence--who-wins-when-settings-collide — the em-dash
  yields the double hyphen, the section-4 "change-x--edit-y" precedent;
  #11-appendix-every-files-functions); every Contents link (header-slug
  and explicit <a name>) resolves.

**Verify-at-build rows (GPR-A duty):**

- ROW compile_mode (F): VERIFIED CORRECT as specced. make_model
  (training.py ~127) defaults compile_mode to "reduce-overhead"; build_specs
  (experiment.py ~1546) setdefaults "default" only for needs_geom (the
  conv / TRF heads). So resmlp -> "reduce-overhead", conv/TRF -> "default".
  Kept verbatim.
- ROW ram_frac (F): CORRECTED. The spec row said only "the parallel sweep
  workers force 0". The tree disagrees for one driver: sweep_ntrain
  (:208), sweep_hyperparam (:419), and bakeoff (:202) force
  worker ram_frac = 0.0 (stream from the one shared dump memmap), but the
  tune / Optuna workers instead DIVIDE data.ram_frac by the worker count
  (tune_single :418-419, `ram_frac / n_workers`), each staging its own
  subset concurrently. The appendix row now states both mechanisms + the
  serial default 0.7. DECLARED correction.

**Deviations declared:**

- The ram_frac row correction above (the tune workers divide, they do not
  force 0).
- A markdown-correctness fix (not a content change): the `model.<head>.`
  `activation` placeholder is written as a single code span so `<head>`
  renders literally instead of being parsed as an HTML tag.
- No behavioral change anywhere (documentation-only; git diff = README +
  this note).

**Gate evidence (GPR-A, raw, Mac):**

- Documentation-only: `git diff --stat HEAD -- emulator/ *.py
  example_yamls/` is EMPTY; the full diff is README.md + this note only.
- Anchors: a slug checker over every Contents [text](#anchor) against the
  header slugs + explicit <a name> anchors -> ALL 74 resolve, including
  the two new ones.
- House scans on the README diff: 0 ` -- `; added all-caps are domain
  acronyms only (YAML / TRF / CUDA / MPS / MLP / IA / CPU); README one-line
  rows exempt from 90-col per file convention.
- Every table row cross-checked against the tree: section A (from_config
  flag>model.activation>H + the head_act pin + the warning, unit B);
  B (validate_phase_block + run_emulator per-pass); C / C2 (resolve_phase_args
  + the freeze_trunk guard, unit A); D (validate_loss both-spellings error);
  E (validate_sweep_paths incl. freeze_trunk, ACTIVATION_PATHS, model.name/ia
  refusal, sweep_ntrain n_train); F (pick_device, DEFAULT_THRESHOLDS, rescale
  flag, ram_frac corrected, compile_mode verified); G (derive_eval_bs,
  run-global bs_base, fixed AdamW/ReduceLROnPlateau, the two TRF pins).
- Whole-tree py_compile clean (documentation-only, run regardless).

Open: the Architect re-audit of GPR-A. No workstation leg (docs only).
This closes the three-unit head-activation cycle.

### 2026-07-07 — Architect re-audit: ACCEPTED, no deltas — CYCLE CLOSED

Own checks on the confirmed base a95f8df:

- Documentation-only PROVEN: `git diff --stat HEAD -- emulator/ *.py
  example_yamls/` empty; the diff is README.md + this note.
- Both verify-at-build rows re-verified IN THE TREE: compile_mode
  correct as specced (make_model default "reduce-overhead",
  training.py 116/127; build_specs setdefault "default" for
  needs_geom, experiment.py 1546); the ram_frac CORRECTION is real —
  tune_single:418-419 divides data.ram_frac by n_workers while
  sweep_ntrain:208 / sweep_hyperparam:419 / bakeoff:202 force 0.0 —
  the appendix row (all three regimes + the serial 0.7 default) is
  now truer than my spec row. Correction ACCEPTED with credit.
- Anchors: my own slug checker first flagged two links, including the
  PRE-EXISTING working section-4 anchor — my slugger collapsed the
  em-dash's two spaces into one hyphen (GitHub keeps both). Fixed my
  checker: all 35 links resolve, none missing. My harness bug, owned;
  the Implementer's 74-anchor claim covered both directions
  (anchors + links), mine checked links only.
- Sections read against the tree (A..G + C2 spot-checked row by row;
  F/G quoted in full); scans 0 double-hyphen; py_compile clean.
- Deviations RULED: the ram_frac correction ACCEPTED (above); the
  `model.<head>.activation` code-span fix ACCEPTED (rendering
  correctness, not content).

UNIT C COMMIT-READY. Suggested sentence: "README precedence appendix
(section 10): who wins when settings collide — activation slots +
pin/alias/license, phase-block semantics, demotion, schedule modes,
loss spellings, sweeps, driver-vs-YAML (ram_frac row corrected
against the tree), deliberately-no-knob; gate GPR-A
Architect-verified". THE THREE-UNIT CYCLE IS CLOSED on my side:
A = ebd9869 (freeze_trunk), B = a95f8df (per-head activation),
C = this commit. Workstation queue gains GFT-C + GHA-F beside the
standing board.
