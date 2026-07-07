---
name: head-activation-per-component
description: "SPEC 2026-07-06 (Architect, handoff ready): per-head-component activation. User asked to change activation {type, n_gates} inside the trunk:/head: PHASE blocks — answer NO, twice: the eight-key phase whitelist rejects it, and it could not work anyway (activation is a construction-time choice; the model is built once before phase 1, the phase blocks are per-pass training diffs, and a mid-run family swap would re-init trained per-feature gamma/beta under frozen weights). The wish is delivered elsewhere: the head is an exact identity through phase 1 and trains only in phase 2, so the head COMPONENT's activation IS the head phase's activation. New optional key model.cnn.activation / model.trf.activation (same {type, n_gates} / bare-string schema as model.activation; absent = share the trunk's, byte-identical); model.activation stays the trunk's + the default. Seam: head_act=None ctor kwarg on the four head classes, falling back to block_opts['act']. Banner free via describe_spec (prints the active head dict verbatim). Sweeps as an ORDINARY dotted leaf (unlike model.activation's ACTIVATION_PATHS special case). Gates GHA-A..F + handoff embedded. NOT implemented."
metadata:
  node_type: memory
  type: project
---

# Per-head-component activation (spec, 2026-07-06)

User question: inside `head:` (and `trunk:`), can I change
`activation: {type, n_gates}`? If not, plan it.

## The answer to the question as asked: no, and no

1. The phase whitelist rejects it. `_PHASE_BLOCK_KEYS` is the eight
   training-loop keys (lr / scheduler / loss / trim / focus / clip /
   rewind / ema); `head: activation:` raises
   `unknown train_args.head key(s): ['activation']`.
2. Whitelisting it could not work. The phase blocks are per-pass
   TRAINING diffs consumed by run_emulator, which never rebuilds the
   model — the network is constructed once, before phase 1, and the
   activation is baked into every block at construction
   (`block_opts["act"]`, experiment.py build_specs ~1285). Worse, the
   activations are LEARNABLE (per-feature gamma / beta are trained
   weights): swapping the family mid-run re-initializes them and
   changes the function under the trained weights — for the trunk,
   exactly at the moment it freezes. Construction knobs are
   run-global; this is the bs_base ruling again (phase blocks are
   diffs, not containers).

## The wish is legitimate and has a clean home

The head is an exact identity through phase 1 (zero-init) and only
trains in phase 2. So "the head phase's activation" and "the head
component's activation" are the same thing — no mid-run swap needed:

- trunk activation = `model.activation` (exists today; it is the
  trunk's family and the default for everything).
- head activation  = NEW optional `activation` key inside
  `model.cnn:` / `model.trf:`; absent = share the trunk's
  (byte-identical to today).

Proposed YAML (paste-ready, the restrf example):

```yaml
  model:
    name:        restrf
    mlp:
      width:    128
      n_blocks: 4
    activation:            # the trunk's family (and the default
      type:    H           # for every component that sets none)
      n_gates: 3
    trf:
      n_heads:      2
      n_blocks:     1
      n_mlp_blocks: 2
      gate_init:    0.1
      activation:          # optional: this head's own family
        type:    gated_power   # (absent = share the trunk's; takes
        n_gates: 3             # effect in phase 2, where the head
                               # trains)
```

## Design (four seams, all verified in the tree at 8bb5484)

1. Model classes — the seam already exists as a local:
   `cnn_act = block_opts.get("act", activation_fcn)` at
   emulator_designs.py:403 (ResCNN) and IA/emulator_designs.py:384
   (TemplateResCNN); `trf_act = ...` at emulator_designs.py:712
   (ResTRF) and IA/emulator_designs.py:797 (TemplateResTRF). Each of
   the four gains a `head_act=None` constructor kwarg; the seam
   becomes `head_act if head_act is not None else
   block_opts.get("act", activation_fcn)`. Docstring Arguments entry
   each. Identity-at-init survives any family (all four map 0 -> 0:
   the gate-times-x forms and psi_p(0) = 0), so the loss-continuous
   handoff is unaffected.

2. build_specs (experiment.py, the MODEL_BLOCK_KEYS translation loop
   at ~1243-1257): add `"activation"` to the cnn and trf tables (NOT
   mlp — `model.activation` IS the trunk's; a second spelling would
   be a trap), and special-case its VALUE before the generic
   `model_opts[table[k2]] = v2` copy: accept the same two shapes as
   the top-level key (mapping {type, n_gates} or a bare type string),
   reject unknown sub-keys loudly (strict {type, n_gates}; note: the
   TOP-level model.activation silently ignores unknown sub-keys
   today — an existing looseness, observed, out of scope), build
   `make_activation(type, n_gates=...)`, set
   `model_opts["head_act"] = factory`. The inactive-head skip already
   ignores the whole cnn:/trf: block on the wrong architecture — the
   nested activation rides along, no new rule. The error message for
   unknown sub-block keys must list activation among allowed.

3. Banner: FREE. describe_spec (emulator_designs.py:75) renders the
   active head block's dict verbatim, so the head activation prints
   in the model-spec line automatically; the env line's
   "activation:" stays the shared/trunk family. Gate asserts both
   (consumed-view directive: the spec states the rendering; this is
   it).

4. Sweeps / tune: the new key sweeps as an ORDINARY dotted leaf
   (`model.trf.activation` with bare-string values, or
   `model.trf.activation.n_gates` numeric) because build_specs reads
   it from train_args — unlike `model.activation`, whose
   ACTIVATION_PATHS special case exists only because from_config
   resolves it onto the experiment. ACTIVATION_PATHS matches exactly
   ("model.activation", "model.activation.type") and does not catch
   the new paths: no driver change. tune: family strings fixed per
   study (the loss.mode rule), n_gates searchable. bakeoff sweeps the
   SHARED family; a pinned head stays pinned — one documented
   sentence in the bakeoff header.

## Rulings

- (a) Precedence — USER-ACCEPTED 2026-07-06 with an amendment: an
  explicit per-head key pins the head; the --activation flag and
  model.activation govern the trunk + default only
  (most-specific-wins). AMENDMENT (user directive): whenever an
  explicit --activation flag is given AND the active head carries a
  pin with a DIFFERENT family, a warning prints in the terminal at
  startup. Mechanism (the resolve_phase_args notice pattern):
  from_config — the one place that knows the flag was explicit (the
  drivers pass None when absent) — compares the flag's type against
  the active head's model.<head>.activation type and stores the
  notice; print_design emits it right after the env line,
  quiet-gated through self.log like the demotion notice. Wording:

      warning: --activation power sets the trunk/default only; the
      head keeps its model.trf.activation pin (gated_power)

  No warning when the flag is absent (a YAML-internal
  model.activation vs head pin is an intentional config, both
  visible in one file) or when the flag and the pin agree (no
  surprise possible). Same-surprise siblings get the same warning
  line, unconditional on equality (their swept values change per
  curve): bakeoff_activation at startup when the active head is
  pinned ("the pinned head stays fixed across the bake-off"), and
  sweep_hyperparam when the swept parameter is in ACTIVATION_PATHS
  and the active head is pinned.
- (b) No `model.mlp.activation` (one spelling per thing).
- (d) LICENSE RULE (user directive 2026-07-06, with
  [[freeze-trunk-joint-phase2]]): a per-head activation — either
  spelling — requires a frozen-trunk head phase: trunk_epochs > 0
  AND freeze_trunk true (on the active head). The head family IS
  the head-phase family; when trunk and head train together
  (freeze_trunk false, or trunk_epochs 0/absent on a Template
  model — the latter an Architect extension of the user's rule,
  veto-able) the network keeps one family. ERROR at build_specs
  (the consumption site), teaching message in the freeze-trunk
  note. Inactive head blocks on resmlp stay ignored as today.
  GHA-A gains the license cases (pin + freeze_trunk false; pin +
  trunk_epochs 0; alias likewise; proper two-phase passes).
- (c) SUPERSEDED 2026-07-06 by user directive — the asymmetry is
  ACCEPTED as a deliberate special case ("this is a special case
  that makes sense to create the asymmetry. Just include a nice
  error message explaining the asymmetry (plus explain in the
  readme)"). The D-L1 lesson applied exactly where it is coherent:

  * `head: activation:` is LEGAL — an alias for the active head
    component's activation. Canonical spelling stays
    model.cnn.activation / model.trf.activation (works on
    single-phase YAMLs too, where the inactive block is ignored).
    Coherent because the head only trains in phase 2. Mechanics:
    consumed by build_specs (where the head factory is built), NOT
    by the per-pass training resolution — validate_phase_block
    accepts "activation" for which == "head" ONLY (the eight
    training keys are unchanged; this is a ninth, head-only,
    non-training key); run_emulator never reads it (docstring line:
    consumed upstream by build_specs; a direct run_emulator caller
    gets no alias). BOTH spellings given (model.<head>.activation
    AND head.activation) -> ERROR naming both, keep one, recommend
    the canonical — the loss berhu:/mode-named precedent: no silent
    winner, even when the values agree. Single-phase demotion drops
    head: whole, activation included, named in the notice
    (consistent: no head exists there). Sweepable as
    head.activation on a two-phase model (ordinary leaf;
    validate_sweep_paths already rejects head.* on single-phase).
    The ruling-(a) flag-vs-pin warning applies to this spelling
    identically — it IS the pin.

  * `trunk: activation:` stays an ERROR — the trunk is the same
    modules in both phases (its activation shapes phase 1 AND the
    frozen phase-2 forward), so a phase-local trunk activation
    cannot exist; the only alias target would be model.activation
    itself. The error message TEACHES the asymmetry:

        train_args.trunk.activation: the trunk is the same modules
        in both phases, so it cannot have a phase-local activation
        — set model.activation (the run's trunk + default family).
        head: activation: IS accepted (the head only trains in
        phase 2); its canonical spelling is model.cnn.activation /
        model.trf.activation.

  * README documents the asymmetry: the two-phase paragraph
    (section 6) + [[readme-precedence-appendix]] (section A gains
    the alias footnote, section B the head-only row).

  GHA-A gains: head alias accepted, factory identical to the
  canonical spelling's; both-spellings error (also when equal);
  trunk error verbatim; demotion drops head.activation with the
  notice.

## Docs owed with the feature (audit-standard)

train_single YAML (commented activation: inside both cnn: and trf:
with the "absent = shares the trunk's; the head trains in phase 2"
note); tune YAML one line; README section 6 model sentence + section
10 emulator_designs entry; train_single driver header clause;
experiment __init__ docstring model entry; MODEL_BLOCK_KEYS comment.
NOTE: the doc-audit fixes (D-DOC1..8, [[audit-docs-2026-07-06]])
touch neighboring text — land that commit FIRST, then this feature
on the clean base.

## Gates (Mac, no torch; exec-extract + stubs)

- GHA-A (translation): build_specs-level cases — dict form, bare
  string, n_gates variants, absent (model_opts carries NO head_act),
  unknown sub-key raises, inactive-head block ignored on resmlp,
  mlp.activation rejected as unknown.
- GHA-B (seam): the four classes accept head_act=None; fallback
  expression exact at all four sites; TRFBlock still receives
  act=trf_act; absent-key path byte-identical (AST or exec-extract).
- GHA-C (static): "activation" in cnn + trf tables only;
  special-case ordered before the generic copy; ACTIVATION_PATHS /
  _PHASE_BLOCK_KEYS / resolve_phase_args untouched.
- GHA-D (banner): describe_spec output contains the head activation
  dict for the active head; drops it for the inactive one. The
  flag-vs-pin warning: rendered when an explicit flag meets a
  differing pin; ABSENT when the flag is absent, when they agree,
  and when no pin exists; every other banner line byte-identical.
  The notice-string builder is a pure helper (flag_type,
  head_block, model_block) -> str | None, exercised case-by-case.
- GHA-D2 (drivers, static): the bakeoff startup warning and the
  sweep_hyperparam ACTIVATION_PATHS warning present at their call
  sites, quiet-gated, one line each.
- GHA-E: house scans on the diff + py_compile + every YAML change as
  a paste-ready block in the report.
- GHA-F (workstation, rides the queue): restrf +
  trf.activation gated_power — banner shows it, head param count
  changes vs H (3K+2 vs 2 per feature in the token MLPs), handoff
  loss-continuous; plus a golden no-key byte-identity run.

## Handoff

### ARCHITECT_HANDOFF
Task: per-head-component activation (spec:
notes/head-activation-per-component.md; read it + [[links]],
especially [[banner-prints-consumed-view]]). Base: the doc-audit
commit (D-DOC1..9) must be landed first; start from that tree.
Scope: the four seams exactly as specced — (1) head_act=None on
ResCNN / ResTRF / TemplateResCNN / TemplateResTRF with the fallback
seam at emulator_designs.py:403/712 + IA:384/797; (2) build_specs
translation ("activation" joins the cnn / trf tables, strict
{type, n_gates} value special-case building make_activation, key
NOT added to mlp); (3) banner: describe_spec covers the spec line
(GHA-D proves it) PLUS the flag-vs-pin warning of ruling (a) — a
pure notice-string helper, from_config detects (it alone knows the
flag was explicit), print_design emits after the env line,
quiet-gated (the demotion-notice pattern); (4) driver changes ONLY
the two warning one-liners (bakeoff startup; sweep_hyperparam
ACTIVATION_PATHS case), nothing else; docs per the note's list +
the warning wording. Rulings: (a) is USER-ACCEPTED as amended (pin
holds + warning); (b)/(c) binding unless vetoed in relay.
Also in scope, per the superseded ruling (c): the head: activation:
alias (validate_phase_block accepts "activation" for which ==
"head" only; build_specs consumes it beside the canonical spelling;
both-spellings error naming both, even when equal; run_emulator
docstring line) and the trunk: activation: asymmetry-teaching error
(wording verbatim in the note); GHA-A gains all four cases, plus
ruling (d)'s license cases (the check reads freeze_trunk /
trunk_epochs at build_specs; message in
[[freeze-trunk-joint-phase2]]). This handoff is a THREE-UNIT cycle,
in order: UNIT A = train_args.freeze_trunk (spec + gates GFT-A/B in
[[freeze-trunk-joint-phase2]]; the unit-B license check reads it, so
A lands first); UNIT B = this feature; UNIT C = the
README precedence appendix per [[readme-precedence-appendix]] —
the Architect's rule inventory is in that note; verify the two
flagged rows against the tree and declare any correction; gate
GPR-A there. The cycle also closes D-DOC2b (three enumeration
sites listed in [[audit-docs-2026-07-06]]'s re-audit verdict): the
run_emulator two-phase diagram legend and the YAML walkthrough
fall to unit A, the train() comment to unit B.
Gates: GHA-A..E incl. D2 on the Mac (recipes above; GHA-F embedded
for the workstation queue) + GFT-A/B for unit A + GPR-A for unit
C. Report: raw gate
outputs, every YAML change as a paste-ready block, deviations
declared. Do not commit: leave the diff uncommitted and print the
suggested commit commands (one per unit).
### END

## Status

SPEC DELIVERED 2026-07-06, NOT implemented. Sequencing: doc-audit
fixes first ([[audit-docs-2026-07-06]]), then this. On the report:
Architect re-audit closes GHA-A..E; GHA-F joins the workstation
board. Suggested commit sentence: "Per-head-component activation:
optional model.cnn/.trf activation {type, n_gates} pins the head's
family (absent = share the trunk's; head trains in phase 2, so this
is the head-phase activation with no mid-run swap; gates GHA-A..E
Architect-verified)".

## Resume state (Implementer appends below)

### 2026-07-06 — Implementer (Opus 4.8): UNIT B executed

Base: unit A committed at ebd9869 (freeze_trunk); started there.
GHA-A..E (incl. D2) pass on the Mac. Commit unit B of the three-unit
cycle; unit C (README precedence appendix) follows after B is committed.

**Done, by seam:**

- Seam 1 (model classes): head_act=None ctor kwarg on ResCNN /
  ResTRF (emulator_designs.py) + TemplateResCNN / TemplateResTRF (IA),
  the seam `cnn_act/trf_act = head_act if head_act is not None else
  block_opts.get("act", activation_fcn)` at all four sites; TRFBlock
  still receives act=trf_act; docstring Arguments entry each.
- Seam 2 (build_specs): "activation" joins the cnn + trf
  MODEL_BLOCK_KEYS tables (-> "head_act"; NOT mlp). Two pure helpers:
  `_head_activation_spec(value, source)` (strict {type, n_gates} or
  bare string) and `_resolve_head_activation(canonical, alias,
  head_block, trunk_epochs, freeze_trunk)` (both-spellings error +
  ruling-(d) license). The loop special-cases k2 == "activation"
  before the generic copy (validates -> head_pin, only the ACTIVE
  head reaches it); post-loop resolves canonical vs the head: alias,
  licenses, and builds `model_opts["head_act"] = make_activation(...)`.
- Seam 3 (banner): describe_spec already renders the active head's
  block verbatim, so the head activation prints free (GHA-D). The
  ruling-(a) warning: `_activation_flag_notice(flag_type, head_block,
  head_pin)` (pure), built in from_config (which captures the explicit
  flag before the block-fallback), stored on exp._activation_notice,
  emitted by print_design after the env line, quiet-gated.
- Seam 4 (drivers): a shared pure `_pinned_head_warning(train_args,
  head_block, what_varies)`; bakeoff logs it at startup; sweep_hyperparam
  logs it in the ACTIVATION_PATHS (act_mode) branch. Both quiet-gated
  through the experiment's log, one warning line each.
- Ruling (c) alias: validate_phase_block accepts "activation" for
  which == "head" only (a ninth, head-only, non-training key; the
  per-pass resolution never reads it); trunk: activation: raises the
  asymmetry-teaching error. run_emulator head_opts docstring notes the
  alias is consumed upstream.
- D-DOC2b obs1: the train() `(lr / loss / trim / focus)` comment
  completed to the eight-key whitelist + the activation alias.
- Docs: train_single YAML (commented activation: in cnn + trf), tune
  YAML (one note), README section 6 + section 10 (ResCNN + ResTRF head
  knobs), train_single driver header, experiment __init__ model entry,
  MODEL_BLOCK_KEYS comment.

**Deviations / declared:**

- The warning helpers take `head_pin` (the resolved pin value from
  EITHER spelling), not the spec's literal `(flag_type, head_block,
  model_block)`. This is strictly more correct: ruling (c) requires the
  head: alias to trigger the ruling-(a) warning too, which a
  model_block-only read would miss. Flag if the Architect wants the
  narrower canonical-only form.
- The two teaching messages (the license error and the trunk-asymmetry
  error) are reproduced verbatim from this note / freeze-trunk EXCEPT
  the one word "IS" -> "is" (house caps rule: no all-caps emphasis in
  strings/comments). Flag if verbatim "IS" is required; trivial to
  restore.
- Interface additions: head_act=None on the four head classes; module
  functions _head_activation_spec / _resolve_head_activation /
  _activation_flag_notice / _pinned_head_warning; exp._activation_notice
  attribute (None on direct __init__). No behavioral change to any
  existing path (absent activation = byte-identical; describe_spec code
  untouched; ACTIVATION_PATHS / _PHASE_BLOCK_KEYS / resolve_phase_args
  unchanged).

**Gate evidence (raw, Mac; no torch — exec-extracted from the tree):**

- GHA-A (translation, pure + loop exec-extract): _head_activation_spec
  (bare string / dict / n_gates default 3 / no-type + unknown-subkey ->
  ValueError / non-str-dict -> TypeError); the real build_specs loop +
  post-resolution over 9 cases (active cnn dict -> head_act factory;
  bare-string trf; absent -> NO head_act; inactive cnn/trf on resmlp
  ignored; model.mlp.activation -> unknown-key error; unknown head
  sub-key raises; a pin with trunk_epochs 0 -> license error end to
  end; head: alias -> head_act; canonical + alias -> error). ALL PASS.
- GHA-A(c/d): _resolve_head_activation both-spellings-even-equal error +
  license (trunk_epochs>0 AND freeze_trunk not False, else the teaching
  error; covers joint + trunk_epochs 0); validate_phase_block accepts
  head: activation:, rejects trunk: activation: with the asymmetry
  message, still rejects an unknown head key. ALL PASS.
- GHA-B (seam, static): head_act=None in all 4 signatures; the fallback
  expression at all 4 sites; act=trf_act intact (2). PASS.
- GHA-C (static): "activation" in cnn + trf tables only; the special
  case ordered before the generic copy; ACTIVATION_PATHS + _PHASE_BLOCK_KEYS
  + resolve_phase_args untouched. PASS.
- GHA-D (banner): describe_spec (exec-extracted) shows the active head's
  activation dict and drops the inactive head (code byte-identical to
  HEAD); _activation_flag_notice rendered only on a differing explicit
  flag vs pin, None when the flag is absent / agrees / no pin / no head.
  PASS.
- GHA-D2 (drivers, static): the bakeoff + sweep ACTIVATION_PATHS
  warnings present at their call sites, quiet-gated through log, one line
  each. PASS.
- GHA-E: 0 py/yaml additions over 90 cols; 0 ` -- `; added caps are
  domain acronyms only (TRF/CNN/MLP/CMB/GG/GI/II + YAML) after de-capping
  six emphasis tokens; the two train_single activation blocks align their
  inline comments at col 30; whole-tree py_compile clean. AST-minus-
  docstrings: the seam files + drivers CODE CHANGED as expected, the
  train_single driver comment-only.

**GHA-F (workstation, rides the queue).** Run from $ROOTDIR on a torch
box, two-phase model, a frozen-trunk head phase:

    R=--root=<root> ; F=--fileroot=<fileroot>
    Y=--yaml=train_single_emulator_cosmic_shear.yaml
    # golden: absent per-head activation is byte-identical pre/post
    git stash && python train_single_emulator_cosmic_shear.py $R $F $Y \
      > /tmp/ha_pre.log 2>&1
    git stash pop && python train_single_emulator_cosmic_shear.py $R $F $Y \
      > /tmp/ha_post.log 2>&1
    diff <(grep -E '^(phase|epoch|best|model spec)' /tmp/ha_pre.log) \
         <(grep -E '^(phase|epoch|best|model spec)' /tmp/ha_post.log)  # EMPTY
    # then a pinned-head run: restrf + ia nla, trunk_epochs: 800,
    # freeze_trunk: true (or absent), and under model.trf:
    #   activation:
    #     type:    gated_power
    #     n_gates: 3
    #   the "model spec:" banner shows the trf activation dict;
    #   the head param count rises vs the shared H (the gated_power
    #   gamma/beta per token MLP feature); the handoff stays
    #   loss-continuous (the zero-init head is identity at any family).
    # flag-vs-pin: add --activation power -> the startup warning prints
    #   "warning: --activation power sets the trunk/default only; the head
    #    keeps its model.trf.activation pin (gated_power)".
    # license: set freeze_trunk: false (or trunk_epochs: 0) with the pin
    #   -> build_specs errors with the frozen-trunk-head-phase message.

Open: GHA-F (workstation) + the Architect re-audit of GHA-A..E. Unit B
commit is independent; unit C (README precedence appendix) follows on
the committed base.

### 2026-07-06 — Architect re-audit: ACCEPTED, no deltas

Own harnesses on the confirmed base ebd9869:

- Footprint exact (AST-minus-docstrings): CODE CHANGED only in
  training.py / experiment.py / the two design files / the two warned
  drivers; train_single driver comment-only; building_blocks, tune,
  sweep_ntrain untouched.
- 30/30 own-harness checks: _head_activation_spec (bare string, dict,
  n_gates default, three rejects); _resolve_head_activation
  (both-spellings error EVEN WHEN EQUAL; the license blocks
  trunk_epochs 0 with freeze_trunk None/True/False AND trunk_epochs>0
  + freeze_trunk False, message teaching); _activation_flag_notice
  full truth table (differing fires, absent/agree/no-pin/no-head
  None, bare-string pin works); _pinned_head_warning (canonical +
  alias spellings, None cases); validate_phase_block asymmetry
  (head: activation accepted, trunk: activation raises the teaching
  error, unknown-key error gains the head-only hint on head and NOT
  on trunk, the eight training keys unchanged at 8).
- Statics: describe_spec and resolve_phase_args AST-identical to
  HEAD; ACTIVATION_PATHS unchanged; all four head_act seams present
  (2 per file pattern) with act=trf_act intact; train() resolves
  phases before build_specs (the alias-only-for-a-real-head claim
  holds); the two >90-col added lines are README one-line rows (the
  file's convention); py_compile clean.
- One harness bug of MINE mid-audit (owned): validate_phase_block's
  internal validate_loss/validate_ema calls needed stubs; re-run
  correctly, all pass. Not a code finding.

Deviations RULED:
- head_pin-based warning-helper signature: ACCEPTED — strictly more
  correct (ruling (c) requires the head: alias to trigger the
  ruling-(a) warning; a model_block-only read would miss it).
- "IS" -> "is" in the two teaching strings: ACCEPTED — the house
  caps rule governs runtime strings; the note's caps were emphasis,
  not content.
- The interface additions (head_act=None x4, the four pure helpers,
  exp._activation_notice) accepted as declared.

UNIT B COMMIT-READY. Suggested sentence: "Per-head-component
activation: model.cnn/.trf activation {type, n_gates} pins the
head's family (head: activation: alias accepted, trunk: errors
teaching the asymmetry; both spellings = error; licensed by a
frozen-trunk head phase — trunk_epochs > 0 + freeze_trunk true;
flag-vs-pin + pinned-head warnings; gates GHA-A..E incl. D2
Architect-verified)". Unit C proceeds on the committed base; GHA-F
rides the workstation queue.

## Proceed directive (Architect, 2026-07-06, after unit A acceptance)

Unit A (freeze_trunk) is Architect-ACCEPTED (re-audit verdict in
[[freeze-trunk-joint-phase2]]: 14/14 + statics, no deltas) and goes
to the user for its commit. Units B (this note) + C
([[readme-precedence-appendix]]) are AUTHORIZED to proceed on the
unit-A committed base:

- Precondition: `git log -1` shows the unit-A sentence
  ("train_args.freeze_trunk (default true): ..."). If the tree still
  carries uncommitted unit-A changes, STOP and report.
- Unit B scope = this note in full: the four seams; ruling (a) as
  amended (pin holds + the flag-vs-pin warning + the bakeoff /
  sweep one-liners); superseded ruling (c) (head: activation: alias
  accepted for which == "head" only, both-spellings error even when
  equal, trunk: activation: asymmetry-teaching error verbatim);
  ruling (d) license at build_specs (trunk_epochs > 0 AND
  freeze_trunk not False — absent counts as true; wording in
  [[freeze-trunk-joint-phase2]]); D-DOC2b obs1 (the train() comment
  enumeration); docs per the note. Gates GHA-A..E incl. D2; GHA-F
  recipe embedded for the workstation queue.
- Unit C after B's commit: the appendix per
  [[readme-precedence-appendix]] including the freeze_trunk rows;
  verify the two flagged rows against the tree, declare any
  correction; gate GPR-A.
- Report per unit (an IMPLEMENTER_HANDOFF after B, another after
  C), resume state appended per unit, raw gate outputs, every YAML
  change as a paste-ready block, deviations declared. Do not
  commit: leave each diff uncommitted and print its suggested
  commit command.
