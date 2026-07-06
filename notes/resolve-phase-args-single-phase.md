---
name: resolve-phase-args-single-phase
description: "Design spec (Architect, 2026-07-06): let one YAML serve both single-phase and two-phase models. Today a train_args with trunk_epochs / trunk: / head: dies in run_emulator when the model lacks set_train_phase (user hit it: name: resmlp + a two-phase YAML -> 'trunk_epochs needs a two-phase model ... this model is OptimizedModule'). Fix: a pure resolve_phase_args(train_args, two_phase) in experiment.py, called once in train() — for a single-phase model it drops head: and trunk_epochs and merges the trunk: block into the top level (the user's rule: 'what is in the trunk is just the global'), with a loud banner notice; for a two-phase model it is an exact no-op. run_emulator's strict guards stay (they protect direct callers); its capability error additionally unwraps _orig_mod so it names the real class, not OptimizedModule."
metadata:
  node_type: memory
  type: project
---

# resolve_phase_args: two-phase YAML keys on a single-phase model

User request 2026-07-06, after the workstation run died: with
`name: resmlp` the two-phase keys should degrade gracefully — "ignore
the head: and assume what is in the trunk is just the global". The use
case is one shared YAML across architectures (the same train_args block
drives rescnn+nla two-phase runs and plain-resmlp baselines).

## Facts (verified in source)

- `run_emulator` (training.py) holds two strict guards: trunk_epochs > 0
  on a model without `set_train_phase` raises (line ~1375, the one the
  user hit), and trunk:/head: blocks with trunk_epochs == 0 raise
  (line ~1241). Both are correct for direct callers and stay.
- Only `TemplateResCNN` and `TemplateResTRF` define `set_train_phase`
  (emulator/IA/emulator_designs.py 528 / 839). `TemplateMLP`
  (resmlp + nla) is also single-phase — so the probe must be the
  capability `hasattr(model_cls, "set_train_phase")` on the class
  (mirroring run_emulator's duck-typed instance check), never the name.
- `experiment.train()` is the single choke point: run / tune / both
  sweeps / bakeoff all funnel through it, and it forwards
  `trunk_epochs` / `trunk` / `head` verbatim (experiment.py ~1138).
- The per-phase override keys are exactly six (run_emulator docstring):
  lr_base, loss_mode, trim, focus, clip, rewind. Top-level asymmetry:
  five have same-named top-level keys, but the top-level base lr lives
  NESTED as `lr: {lr_base, bs_base, warmup_epochs}` while the phase key
  is a bare `lr_base`.
- The guard's error message names the compiled wrapper
  (`this model is OptimizedModule`) instead of the real class — a
  diagnostics wart to fix in the same pass.

## Design

1. **Pure `resolve_phase_args(train_args, two_phase)`** in
   experiment.py (beside validate_sizes; no torch). Returns
   `(resolved_train_args, notice)`:
   - `two_phase=True` -> the input returned unchanged, notice None
     (an exact no-op: two-phase runs must be byte-identical to today).
   - `two_phase=False` and none of trunk_epochs / trunk / head present
     -> unchanged, notice None (plain single-phase YAMLs untouched).
   - `two_phase=False` otherwise -> a copy (never mutate the input:
     sweep drivers reuse self.train_args across points) with:
     - `head` dropped, `trunk_epochs` dropped, `trunk` dropped;
     - each key of the old trunk: block merged INTO the top level,
       full-replacement (the same semantics the block has as a phase
       override — trim / focus replace whole blocks, no deep merge):
         loss_mode -> train_args["loss_mode"]
         trim      -> train_args["trim"]
         focus     -> train_args["focus"]
         clip      -> train_args["clip"]
         rewind    -> train_args["rewind"]
         lr_base   -> train_args["lr"]["lr_base"] (copy the lr block,
                      preserve its bs_base / warmup_epochs; create the
                      block if absent)
     - an unknown key inside trunk: raises ValueError naming the six
       allowed (the usual typo guard);
     - notice = one line naming what actually happened, e.g.
       `single-phase model: trunk: merged into the top level
       (loss_mode); head: and trunk_epochs ignored` — list only the
       keys really merged, and only the parts really present (a YAML
       with head: but no trunk: says "head: ignored" alone).
2. **Call site**: top of `train()`, right after the train_args default
   resolution:
   `hasattr(self.model_cls, "set_train_phase")` -> two_phase; log the
   notice through self.log (quiet-gated, like the config banner). This
   covers run(), tune, sweep_hyperparam, sweep_ntrain, bakeoff in one
   place. build_specs and run_emulator then see clean args.
3. **training.py wart fix**: the capability guard's message names
   `type(getattr(model, "_orig_mod", model)).__name__` instead of the
   wrapper. Both run_emulator guards otherwise stay verbatim (direct
   callers still fail loudly; the experiment path simply no longer
   sends them dirty args).
4. **Docs**: run_emulator's trunk_opts/head_opts docstring gains one
   line ("EmulatorExperiment.train resolves these away for
   single-phase models; direct callers pass clean args");
   experiment.__init__'s train_args docstring documents the demotion
   rule; the train_single example YAML's trunk:/head: comment notes
   the blocks are ignored-with-notice on single-phase models. No YAML
   key changes (nothing to migrate).

Documented consequence (not a bug): sweeping a head.* key with a
single-phase model makes every sweep point identical — the notice line
is the tell. Out of scope: trunk:/head: key whitelisting on the
two-phase path (today unvalidated; separate item if wanted).

## Validation gate

- GP-A (pure, Mac, exec-extracted resolve_phase_args): (1) two_phase
  identity incl. input-unmutated; (2) the user's exact failing block
  (trunk {loss_mode: sqrt}, full head:, trunk_epochs) -> head +
  trunk_epochs gone, loss_mode merged, TOP-LEVEL trim/focus survive,
  notice names loss_mode only; (3) all six keys incl. lr_base -> lr
  block updated, bs_base/warmup preserved, lr block created if absent;
  (4) no two-phase keys -> unchanged, no notice; (5) unknown trunk key
  -> ValueError naming the six; (6) head-only YAML -> dropped + notice,
  nothing merged; (7) deep-unmutated input in every case (snapshot
  compare). Paste raw outputs.
- GP-B (static, Mac): AST check that train() calls resolve_phase_args
  before run_emulator and probes self.model_cls; the training.py
  message now unwraps _orig_mod; run_emulator's two guards unchanged.
- GP-C style: house scans on touched files; whole-tree py_compile.
- GP-D (workstation, rides the queue): (1) the exact YAML + name:
  resmlp that produced the traceback now trains, banner shows the
  notice; (2) control: the same YAML on rescnn+nla reproduces today's
  behavior exactly (resolution is a no-op for capable models).

## Sequencing

Stacks on the uncommitted n_train/n_val feature
([[n-train-n-val-absolute-counts]], delta D-1 still open). Implementer:
close D-1 first (one guard line), then this feature. Two separate
commit units for the user:

    git add emulator/data_staging.py notes/n-train-n-val-absolute-counts.md
    git add emulator/experiment.py example_yamls/ sweep_hyperparam_emulator_cosmic_shear.py train_single_emulator_cosmic_shear.py README.md
    git commit -m "Replace train/val divisors with absolute n_train/n_val enforced after param_cuts (loud migration, gates GS-A-C Architect-verified)"
    git add -A
    git commit -m "Resolve two-phase YAML keys on single-phase models: trunk: becomes global, head:/trunk_epochs ignored with notice (gates GP-A-C Architect-verified)"

(exact file split to be confirmed at acceptance; then `cd ../../..` and
`git merge claude/amazing-keller-e798b6`.)

## Resume state (Implementer appends below)

Implemented 2026-07-06 (Opus 4.8, amazing-keller); execution log + raw
gate evidence in the last section of this file.

### ARCHITECT_HANDOFF: READY FOR EXECUTION

- **Notes entry (read first):** this file.
- **Target file(s):** emulator/experiment.py (resolve_phase_args +
  train() call site + __init__ docstring), emulator/training.py (the
  _orig_mod unwrap in the capability error; guards otherwise verbatim),
  example_yamls/train_single_emulator_cosmic_shear.yaml (one comment on
  the trunk:/head: blocks).
- **Contracts & interfaces:** `resolve_phase_args(train_args,
  two_phase)` pure, returns `(resolved, notice_or_None)`; never mutates
  the input; exact no-op for two_phase=True. The six-key mapping table
  above is verbatim numerics (note the lr_base -> lr.lr_base nesting).
  This adds a function to experiment.py's public surface — any further
  interface deviation must be declared.
- **Constraints & edge cases:** capability probe on the CLASS
  (TemplateMLP is single-phase too); notice lists only what actually
  happened; trim/focus merge is full replacement, never deep; unknown
  trunk key raises; head contents never validated, only dropped;
  run_emulator's guards must not be weakened.
- **Validation gate:** GP-A/B/C on the Mac, raw outputs pasted; GP-D
  rides the workstation queue (now: G-F, GN-F, GT-B/GT-C, GS-D, GP-D,
  item-27, G1 import leg).
- **Next milestone:** IMPLEMENTER_HANDOFF with GP-A/B/C evidence,
  after D-1 of [[n-train-n-val-absolute-counts]] is closed in the same
  session.

### 2026-07-06 — Implementer (Opus 4.8) execution

D-1 of [[n-train-n-val-absolute-counts]] closed first this session (the
load_source keep<1 guard + GS-B -5/0 cases), then this feature. Mac dev
box: GP-A/B/C ran by exec-extraction / AST as usual; GP-D rides the
workstation queue.

**Done (exactly the handoff's target-file list):**

- experiment.py: new standalone pure `resolve_phase_args(train_args,
  two_phase)` beside validate_sizes (+ constant `_PHASE_TRUNK_KEYS`, the
  six per-phase override keys). two_phase or no-phase-keys -> the input
  returned unchanged, notice None. Single-phase with phase keys -> a
  SHALLOW copy (input never mutated; the lr block is copied before
  updating lr_base) with head / trunk_epochs / trunk dropped and each
  present trunk: key merged into the top level by the six-key table
  (lr_base -> lr.lr_base preserving bs_base / warmup_epochs; the other
  five full-replacement); an unknown trunk: key raises ValueError naming
  the six; the notice names only what really happened. Called at the top
  of `train()` (the choke point for run / tune / both sweeps / bakeoff),
  probing `hasattr(self.model_cls, "set_train_phase")` on the CLASS and
  logging the notice through self.log (quiet-gated). __init__ train_args
  docstring documents the demotion rule.
- training.py: the capability guard's message now names
  `type(getattr(model, "_orig_mod", model)).__name__` (the real class,
  not the OptimizedModule compile wrapper); the guard CONDITION and both
  other phase guards are byte-identical to HEAD (GP-B verified). The
  trunk_opts / head_opts docstring gains one line: train() resolves these
  away for single-phase models, direct callers pass clean args.
- example_yamls/train_single YAML: one comment on the trunk: / head:
  blocks noting they are ignored-with-notice (trunk: merged to global) on
  a single-phase model, so one YAML drives both families.

**Deviations from blueprint:** none. resolve_phase_args is the one added
public function (declared in the contract). The changed-file set is
exactly the three target files (plus the D-1 one-liner in data_staging.py,
which belongs to the n_train/n_val commit).

**Gate evidence (raw, Mac):**

- GP-A resolve_phase_args (pure, exec-extracted; every case deep-snapshots
  the input and re-compares after): (1) two_phase -> same object, notice
  None, unmutated; (2) the user's exact failing block (trunk
  {loss_mode: sqrt} + full head: + trunk_epochs 1500) -> head +
  trunk_epochs + trunk dropped, loss_mode merged, TOP-LEVEL trim / focus
  survive, lr untouched, notice = `single-phase model: trunk: merged into
  the top level (loss_mode); head: and trunk_epochs ignored` (names
  loss_mode only, no trim/focus claim); (3) all six keys incl lr_base ->
  lr.lr_base updated, bs_base/warmup preserved, trim/focus/clip/rewind
  full-replaced, input lr.lr_base still 0.005; (3b) lr block created when
  absent; (4) no phase keys -> same object, no notice; (5) unknown trunk
  key -> ValueError naming the offender + the six; (6) head-only ->
  `single-phase model: head: ignored`, nothing merged; input unmutated in
  all cases. GP-A: ALL PASS.
- GP-B static wiring (AST): train() calls resolve_phase_args before
  run_emulator, probes hasattr(self.model_cls, "set_train_phase"), passes
  two_phase=, logs via self.log; training.py uses
  getattr(model, "_orig_mod", model); all three run_emulator phase-guard
  conditions byte-identical head vs work. GP-B: ALL PASS.
- GP-C style: 0 new all-caps / 0 new ` -- ` on the three touched files;
  experiment.py + training.py <= 90 cols; 0 new comprehensions (0->0 both
  modules); whole-tree py_compile clean. GP-C: ALL PASS.
- GP-D workstation: NOT run (no torch/cosmolike on the Mac). (1) the exact
  YAML + name: resmlp that produced the traceback must now train, banner
  showing the notice; (2) control: the same YAML on rescnn+nla must
  reproduce today's behavior exactly (resolution is a no-op for capable
  models). Rides the workstation queue.

Open: GP-D only, plus the Architect re-audit of this feature.

### 2026-07-06 — Architect re-audit: ACCEPTED with ONE doc delta (D-P1)

Verified independently (own harness, own cases, exec-extracted
resolve_phase_args + the real data_staging module): every claim
reproduced. GP-A: the user's exact failing block resolves with top-level
trim/focus surviving and the notice naming loss_mode only; all-six merge
updates lr.lr_base preserving bs_base/warmup (input lr block unmutated,
block created when absent); unknown trunk key raises naming the six;
head-only / trunk_epochs-only / empty-trunk notices exact; identity paths
return the same object. GP-B: all three run_emulator guard conditions
byte-identical vs HEAD; the message unwraps _orig_mod; train() wiring is
resolve -> build_specs -> run_emulator probing self.model_cls and logging
via self.log. GP-C: my scans + whole-tree py_compile clean. D-1 also
verified: -5 and 0 raise the guard; the 700-exact, pool-too-small, and
nesting regressions still pass. Code deviations: none.

**D-P1 (required, doc-only): fix the single-phase examples.** Three
added doc lines use misleading examples/jargon:
- experiment.py __init__ docstring: "e.g. resmlp / nla" — `nla` is not
  a single-phase marker (rescnn + nla = TemplateResCNN IS two-phase);
- experiment.py train() comment: "e.g. resmlp or nla" — same problem;
- the YAML comment: "or an ia without a factored head" — conflates
  factored with two-phase (TemplateMLP is factored AND single-phase;
  the capability that matters is set_train_phase).
Correct wording in all three spots: single-phase = any model class
without set_train_phase — every `name: resmlp` (including
`ia: nla / tatt`, the factored TemplateMLP); two-phase = rescnn / restrf
with an ia (TemplateResCNN / TemplateResTRF). No code change; re-run the
scans + py_compile on the two files.

Noted, not a delta (bundled with the out-of-scope two-phase whitelist
item): a scalar `trunk: sqrt` on a single-phase model is silently
dropped with the notice "trunk: ignored (no keys)" — mildly misleading,
but the two-phase path has the same unvalidated-block looseness today;
one future item should make both loud together.

Process correction to the Implementer's D-1 closure line ("folds into
the n_train/n_val commit unit"): 906528c is already published on main,
so D-1 is its OWN follow-up commit — the Implementer's handoff said this
correctly; the note line was stale. Commit split at acceptance:
D-1 = data_staging.py + its note (ready now); resolve_phase_args =
experiment.py + training.py + the YAML + its note + MEMORY.md (after
D-P1).

### 2026-07-06 — D-P2 (user-found bug): the banner is not
### capability-aware

User report: with name: resmlp (TemplateMLP, single-phase) and
nepochs 1000 / trunk_epochs 2000, print_design printed
`(two-phase: 2000 trunk + -1000 head)` — a negative head count for a
run that will train single-phase. print_design computes the split
from the RAW train_args and never probes the model; the demotion
feature made train() capability-aware but not the banner (an
Architect audit gap: GP-B checked train(), not print_design).

Fix (experiment.py, print_design only): resolve first, print the
truth —
1. probe two_phase = hasattr(self.model_cls, "set_train_phase") (the
   same probe train() uses);
2. call resolve_phase_args(self.train_args, two_phase) — pure and
   non-mutating, so the banner reuses the real machinery instead of
   duplicating its logic;
3. print every block line from the RESOLVED args: on a single-phase
   model the two-phase fragment and the trunk:/head: block lines
   disappear, the merged values appear in the top-level lines, and
   the resolution notice is printed as part of the banner (the
   train-time repeat of the notice is harmless; sweeps run quiet);
4. two-phase models print exactly as today (resolution is a no-op).

Gate D-P2: for the user's exact config (resmlp + nepochs 1000 +
trunk_epochs 2000 + trunk berhu / head blocks) the banner contains NO
two-phase fragment and NO negative number, shows the trunk-merged
loss in the run line, and carries the notice; a rescnn+nla config
prints byte-identically to today (golden, string compare of the
banner); a two-phase config with trunk_epochs >= nepochs still dies
on run_emulator's existing guard (banner does not pre-judge it).
Folds into the pending unit (anneal + D-L1v3).

### 2026-07-06 — D-P2 extended to D-P2v2 (adds D-P3; the design follows
### the user's class-prints-itself tip and the standing directive
### [[banner-prints-consumed-view]])

Leg (a), phases — unchanged from D-P2: print_design resolves via the
pure resolve_phase_args + the capability probe and prints the resolved
view (no two-phase fragment / trunk: / head: lines on a single-phase
model; the notice joins the banner; two-phase unchanged).

Leg (b), model spec (D-P3, user report: raw cnn:/trf: blocks printed
under TemplateMLP): each design class gains a `head_block` class
attribute (None for ResMLP/TemplateMLP, "cnn" for the CNN pair, "trf"
for the TRF pair) and a shared `describe_spec(model_block)`
classmethod rendering ONLY name / ia / mlp / activation / the class's
own head block / compile_mode. print_design's line becomes the pure
delegation `self.model_cls.describe_spec(ta["model"])`. ARCH_HEAD is
RETIRED: build_specs and build_geometry read model_cls.head_block
(one source of head-knowledge; a new architecture without head_block
fails at class definition, not silently in a banner). build_specs'
tolerant consumption (inactive head blocks ignored, active-block
typos loud) is behavior-unchanged.

Gate D-P2v2: (a) the user's resmlp config — banner has no two-phase
fragment / negative number, carries the notice; rescnn+nla banner
unchanged EXCEPT the model-spec line, asserted separately: (b) on
TemplateMLP the spec line contains mlp/activation/compile_mode and NOT
cnn/trf; on TemplateResCNN it contains cnn and not trf; on ResTRF trf
and not cnn/ia; every design class defines head_block; ARCH_HEAD has
zero remaining consumers (grep); build_specs behavior byte-identical
(inactive-ignore + active-typo-raise cases re-run); scans + whole-tree
py_compile. Folds into the pending unit (D-L1v3, closed, awaits this).

### 2026-07-06 — Implementer (Opus 4.8): D-P2v2 executed (BOTH legs)

Process finding (flagged): the handoff says leg (a) is "unchanged from
D-P2", but D-P2's banner code was never committed -- HEAD 5b65fd5's
"capability-aware banner (D-P2)" was a note-only commit (its
c328da9->5b65fd5 diff and print_design both confirm print_design still read
the RAW train_args). So I implemented BOTH legs.

**Leg (a) -- print_design (experiment.py):** probe
`two_phase = hasattr(self.model_cls, "set_train_phase")`, resolve
`ta, notice = resolve_phase_args(self.train_args, two_phase)` (the pure,
non-mutating function train() uses), print the notice when non-None, and
drive every downstream line off the resolved ta -- a single-phase model
that carried two-phase keys now prints them demoted (no two-phase fragment,
no negative head count, no trunk:/head: lines); a two-phase model resolves
to a no-op and prints as before. Docstring updated to the consumed-view
contract.

**Leg (b) -- the class prints itself:** a `DesignSpec` mixin
(emulator_designs.py) provides `head_block` (enforced at class-definition
via __init_subclass__ -- a new architecture that forgets it fails at
import) + a shared `describe_spec(model_block)` classmethod rendering only
name / ia / mlp / activation / this class's own head / compile_mode. All
six design classes mix it in: ResMLP / TemplateMLP head_block=None, the CNN
pair "cnn", the TRF pair "trf" (== the old ARCH_HEAD per architecture). IA
classes import DesignSpec from ..emulator_designs. print_design's spec line
is now `self.model_cls.describe_spec(ta["model"])`. ARCH_HEAD is DELETED;
build_specs reads `self.model_cls.head_block` (the one line changed; the
skip + typo-raise loop is textually unchanged, so its behavior is
byte-identical); the from_config comment updated.

**Gate D-P2v2 (raw, Mac -- exec-extract, no torch):**

    leg (a) capability-aware banner .... 5/5 OK  (single-phase demotes:
      trunk_epochs 0, head count 1000 not -1000, trunk/head lines gone,
      notice present; two-phase resolve is a no-op -> banner unchanged)
    leg (b) describe_spec + head_block .. 4/4 OK  (MLP shows mlp/activation/
      compile_mode not cnn/trf; CNN shows cnn not trf; TRF shows trf not
      cnn/ia; a class missing head_block fails at class definition)
    static: six classes + retirement .... 15/15 OK  (each class head_block
      == old ARCH_HEAD + mixes in DesignSpec; ARCH_HEAD gone; build_specs
      reads head_block; print_design delegates + resolves + prints notice)
    build_specs loop (unchanged) ........ 2/2 OK  (resmlp inactive-ignore:
      a cnn: typo is skipped, not raised; rescnn active-typo: the cnn: typo
      raises model.cnn.bogus_key)

    D-P2v2 gate: ALL PASS

    House scans: 0 over-width, 0 new ` -- `, new all-caps = {IA, YAML}
    (allowlisted acronyms; de-capped CONSUMED / THIS / MRO in the added
    docstrings). Whole-tree py_compile OK.

**Commit:** the handoff expected a fold-in, but D-L1v3 is already committed
(5b65fd5). So D-P2v2 lands as its own follow-up commit (files:
emulator/experiment.py, emulator/emulator_designs.py,
emulator/IA/emulator_designs.py, this note + banner-prints-consumed-view.md
+ MEMORY.md). Awaits the Architect re-audit.

### 2026-07-06 — Architect: D-P2v2 verified CLOSED (both legs)

Independently verified (own harness, 18 checks, all PASS; display
surfaces read directly per the standing directive). Leg (a): the
user's exact config resolves to no two-phase fragment / no negative
number with the notice present; a two-phase model keeps the fragment
(no-op resolution); print_design is ordered probe -> resolve ->
describe, the fragment is gated on the RESOLVED trunk_epochs, and the
block lines print from the resolved ta (trunk/head vanish on
single-phase). Leg (b): DesignSpec.describe_spec exercised with fake
classes on the user's exact model block (None-head shows no cnn/trf;
cnn-head shows cnn not trf; trf-head the reverse);
__init_subclass__ raises at class definition when head_block is
missing (the fail-at-import guarantee); all six design classes declare
the correct head_block; ARCH_HEAD has zero references anywhere;
build_specs' diff is exactly one line (the head lookup), so the
tolerant-consumption behavior is unchanged by construction. Scans +
whole-tree py_compile clean.

The unit (D-L1v3 + D-P2v2) is complete. Commit (user):

    git add -A
    git commit -m "Accept the mode-named berhu block (D-L1v3) + consumed-view banner via class-owned describe_spec, ARCH_HEAD retired (D-P2v2); Architect-verified"
