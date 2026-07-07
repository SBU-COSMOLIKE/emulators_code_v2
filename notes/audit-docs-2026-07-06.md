---
name: audit-docs-2026-07-06
description: "Architect documentation audit at 8bb5484 (README + in-file docs + example YAMLs + driver headers) against the nine 2026-07-06 feature commits. VERDICT: function-level docstrings written this session are accurate everywhere checked; zero retired names (divisor / flat loss_mode / ARCH_HEAD) survive in any user-facing doc; the train_single example YAML is fully current except one line. THREE real problem classes: D-DOC1 (CODE, consumed-view directive violation: print_design's sub-block tuple omits loss and ema, so berhu knot/cap/anneal and a top-level ema: never print at startup while the body comment claims the whole resolved config is on the terminal), D-DOC2 (enumeration rot: five sites still say 'seven keys' or enumerate phase-override keys without ema/scheduler), D-DOC4..6 (README: EMA absent entirely, berhu reduced to one parenthesis, anneal_value described as trim/focus-only, appendix function inventories missing the whole validator layer + derive_eval_bs, no single-phase demotion story, and the BerHu naming caveat documentation duty unpaid). Fix plan + gates GDOC-A..E + the handoff embedded."
metadata:
  node_type: memory
  type: project
---

# Documentation audit (README + in-file), 2026-07-06, at 8bb5484

User request: "do an in-depth audit on the documentation (because we
did a lot of new updates in the code) - both the readme and also
in-file documentation." Audit target: main == amazing-keller ==
8bb5484 (merged; verified via git worktree list). Every finding below
was read from the tree by the Architect, not delegated.

Surfaces read: README.md (all 902 lines); module docstrings +
key function docstrings of training.py, loss_functions.py,
experiment.py, batching.py, emulator_designs.py, data_staging.py,
IA/emulator_designs.py, IA/loss_functions.py; all three
example_yamls; all five driver headers; sweep driver whitelists;
print_design body; texnotes (out of scope: analytic-R only).

## What is CLEAN (verified, no action)

- Zero retired names in any user-facing doc: no train_divisor /
  val_divisor, no flat loss_mode-as-key, no ARCH_HEAD anywhere
  (the only divisor / loss_mode text left is the migration-message
  code itself, which is correct).
- Function-level docstrings from this session's features are
  accurate and complete everywhere checked: run_emulator (loss /
  ema / the eight phase keys), training_loop_batched (ema + berhu
  args in full), eval_val (documents derive_eval_bs), the whole
  validator family, resolve_phase_args, validate_sweep_paths,
  CosmolikeChi2.loss / _reduce (full mode ladder + berhu_s),
  print_design's consumed-view prose, EmulatorExperiment.__init__
  (loss incl. mode-named-block acceptance; ema incl. anneal +
  null opt-out; demotion; n_train / n_val).
- data_staging.py: module docstring + load_source fully current
  (n_keep post-cut, pool-too-small raise).
- IA files: factored losses document the forwarded berhu_knot /
  berhu_cap / berhu_s kwargs at both sites.
- example train_single YAML: every feature block documented
  (loss + berhu + anneal, ema + anneal + per-phase + null note,
  nested phase lr, bs_base run-global note, demotion note, the
  derived eval-bs comment on bs) except the one line in D-DOC2.
- sweep + tune YAMLs current (sweep rules incl. single-phase
  rejection; tune's loss.mode fixed-string + berhu search-leaf
  notes). SWEEPABLE_TOP_KEYS carries loss + ema.

## Deltas

### D-DOC1 (CODE, the one directive violation) CONFIRMED

print_design's sub-block loop (experiment.py, in the tail of
print_design) prints ("optimizer", "lr", "scheduler", "trim",
"focus", "trunk", "head") and its comment claims "so the whole
resolved config is on the terminal". Missing: "loss" and "ema".

Consequences: a top-level ema: block NEVER appears in the startup
banner (the loop's own "ema: horizon ..." print at training.py:1382
fires only after staging + geometry, minutes later); the berhu
knot / cap / anneal sub-block prints NOWHERE at startup (the run
line shows only the bare mode string). Per-phase loss / ema escape
only because trunk: / head: dump their whole dict. This is the
banner-prints-consumed-view directive violated by two features
(loss nesting dcbaf9f, ema 7ebc061/8bb5484) whose banner rendering
the audits under-checked; the Architect owns the audit gap (the
same class as the owned GP-B gap).

Fix (additive-only, so the pending workstation gate recipes that
grep existing banner text stay valid): insert "loss" and "ema"
into the tuple, order ("optimizer", "lr", "scheduler", "loss",
"trim", "focus", "ema", "trunk", "head"); update the docstring's
lines-printed diagram to match. Existing lines stay byte-identical.
DEFERRED (post gate board): the run line's "loss_mode {mode}" label
predates the loss: nesting; renaming it to "loss {mode}" would
invalidate recorded gate recipes, so it waits.

### D-DOC2 (enumeration rot: seven-vs-eight, five sites)

_PHASE_BLOCK_KEYS has EIGHT keys (ema joined in 8bb5484); the
run_emulator docstring and training.py:219 correctly say eight.
Stale sites, each to say eight and list ema (and scheduler where
missing):

1. training.py:279 validate_phase_block docstring: "the seven-key
   whitelist"; same docstring ~line 263 "the other five keys"
   (now six).
2. training.py:2063 comment: "the seven-key whitelist".
3. training.py:2270 comment: "The seven keys mirror the top-level
   schema" + an enumeration missing ema.
4. experiment.py __init__ docstring (trunk / head entry, ~line
   630): "(lr / loss / trim / focus / clip / rewind)" misses
   scheduler AND ema.
5. example_yamls/train_single_emulator_cosmic_shear.yaml:147.
   Paste-ready replacement (comment block, cols preserved):

```yaml
  # Per-phase overrides, symmetric (both need trunk_epochs > 0; error
  # otherwise). Each block mirrors the top-level train_args schema
  # key-for-key; the eight keys are lr / scheduler / loss / trim /
  # focus / clip / rewind / ema, each optional and falling back to
  # the run default. lr is an overlay {lr_base, warmup_epochs} onto the
```

   (line 149's continuation "lr: block (bs_base stays run-global..."
   follows unchanged.)

### D-DOC3 (module docstrings: stale inventories)

- training.py module docstring: no mention of the validation /
  derivation layer (validate_phase_block, validate_loss,
  validate_berhu, validate_ema, derive_eval_bs, derive_ema_beta;
  lines 234..1005, most of the file's top half), of EMA in the
  loop summary, or of the loss modes. One added paragraph naming
  the layer + "trim / focus / berhu-blend / EMA annealing,
  best-epoch tracking (on the EMA average when ema: is set)".
- loss_functions.py module docstring: CosmolikeChi2 summary says
  "(trimming, a focal hardness weight, a sqrt / pseudo-Huber
  transform)" (misses berhu / berhu_capped); "anneal_value is the
  per-epoch trim / focus schedule" (now FOUR consumers: trim,
  focus, the berhu blend s, the EMA horizon).

### D-DOC4 (README: EMA absent entirely)

grep "EMA|ema|horizon" over README.md: zero hits. Needed: one
sentence in section 2 step 7 (selection on the average when ema:
is set); the ema block in the section 6 YAML tour sentence; entries
in the section 10 training.py appendix (validate_ema,
derive_ema_beta, and the loop entry gaining "optional Polyak
weight average coupled to the best snapshot / rewind").

### D-DOC5 (README: berhu layer + the naming-caveat duty)

- Line 344 table: "anneal_value (trim/focus schedule)" stale (see
  D-DOC3).
- Line 790 (loss appendix): CosmolikeChi2 "(trim / focus / sqrt
  transform)" misses the mode ladder.
- The loss appendix (section 10 loss_functions.py) gets the mode
  list + berhu {knot, cap, anneal} one-liner AND pays the
  documentation duty from the berhu naming ruling
  ([[loss-mode-berhu]]): state that this is exact textbook BerHu
  in the whitened residual norm with delta = sqrt(knot), applied
  per sample (Mahalanobis aggregate), and that knot / cap are in
  chi2 units, not residual units.

### D-DOC6 (README: appendix inventories + section 6 gaps)

- Section 10 training.py appendix missing: validate_phase_block,
  validate_loss, validate_berhu, validate_ema, derive_eval_bs,
  derive_ema_beta (README's own rule: one line per function).
- Section 10 experiment.py appendix missing: validate_param_cuts,
  validate_sizes, resolve_phase_args, validate_sweep_paths;
  print_design's line gains "renders the consumed view (the model
  describes itself via describe_spec)".
- Section 6: the two-phase sentence names trunk: / head: but not
  the eight keys, the nested phase lr, per-phase scheduler / loss
  / ema, or the single-phase demotion (only the sweep table row
  hints at it); nothing anywhere says the eval batch is derived
  (~1024 target) and independent of bs. Two or three sentences.
- Section 3 training.py row: add the validator layer clause.

### D-DOC7 (driver header: train_single)

The train_single header's train_args enumeration (lines ~53-73)
has NO ema key at all, and its phase-override list "(lr / loss /
trim / focus / clip / rewind)" misses scheduler + ema. The other
four drivers defer to this header ("see the training driver's
header for every key"), so fixing it fixes the family.

### D-DOC8 (low, batch of nits)

- tune YAML: add ema.horizon_epochs / ema.anneal.* to the
  "legitimate numeric search leaves" note (the loss.berhu note is
  the precedent).
- emulator_designs.py module docstring: no mention of DesignSpec /
  head_block (the class-describes-itself machinery); one sentence.
- experiment.py __init__ "model_cls = the model class (ResMLP /
  ResCNN)" omits ResTRF.

### D-DOC9 (user ruling 2026-07-06: the TRF MLP width pin, explicit)

Every MLP layer inside a TRFBlock runs at the token width (dim =
the padded bin length, 26 on LSST-Y1): the "tokens at their natural
width" decision pinned the TOKEN width to physics, and the per-token
MLP interior inherits that pin — there is deliberately no interior
width knob (n_mlp_blocks sets depth only). The user ruled the pin a
fair rule and vetoed-for-now the mlp_width knob
([[trf-mlp-width-knob]], SHELVED); the documentation must state the
rule EXPLICITLY wherever the trf knobs are described:

- emulator_designs_building_blocks.py TRFBlock docstring: the
  n_mlp_blocks Arguments entry gains "every layer runs at the token
  width (dim -> dim); the interior width is pinned to the bin
  length by design, no width knob" (+ the MLP-branch construction
  comment says dim -> dim).
- emulator_designs.py ResTRF + IA/emulator_designs.py
  TemplateResTRF: the head-knob descriptions gain the same clause.
- experiment.py __init__ docstring trf entry: extend "the tokens
  live at the natural bin width, so there is no width knob" with
  "and the per-token MLP layers run at that width too (n_mlp_blocks
  is depth only)".
- README: the section 6 trf sentence and the section 10 TRFBlock +
  ResTRF entries.
- example train_single YAML: the n_mlp_blocks comment line; tune
  YAML: the trf comment.
- train_single driver header: the trf enumeration clause.

## Gates (fix verification, Mac-runnable, no torch)

- GDOC-A: exec-extract print_design with a stub self.log; feed a
  resolved ta carrying loss {mode berhu_capped, berhu {knot, cap,
  anneal}} + ema {horizon_epochs, anneal} + all prior blocks;
  assert one "loss: {...}" line and one "ema: {...}" line appear,
  AND that every previously printed line is byte-identical to the
  pre-fix output (gate-recipe safety).
- GDOC-B: tree-wide grep: "seven-key" and "seven keys" return
  nothing; the five D-DOC2 sites each list all eight keys.
- GDOC-C: both module docstrings name berhu + berhu_capped and all
  four anneal_value consumers; training.py's names the validator
  layer.
- GDOC-D: README greps: ema present in sections 2/6/10; derive_eval_bs
  present; the anneal_value table row updated; the BerHu
  whitened-norm caveat sentence present; the demotion sentence in
  section 6; the D-DOC9 token-width-pin clause present at every
  listed surface (grep "token width" beside n_mlp_blocks in each).
- GDOC-E: house scans on the diff (90 cols, no double-hyphen
  em-dash spacing, caps allowlist RAM/YAML/GPU/EMA/CPU/CUDA/MPS/
  VRAM, no non-hot comprehensions); YAML hash-column alignment on
  every touched comment; py_compile on touched modules; the
  Implementer report shows EVERY YAML edit as a paste-ready block
  (user standing rule).

## Handoff

### ARCHITECT_HANDOFF
Task: documentation-truth fixes from the 2026-07-06 doc audit
(note: notes/audit-docs-2026-07-06.md; read it + the [[links]]
it cites, especially [[banner-prints-consumed-view]] and the
berhu naming ruling in [[loss-mode-berhu]]).
Scope, in order:
1. D-DOC1 (code): add "loss" and "ema" to print_design's
   sub-block tuple, order ("optimizer", "lr", "scheduler",
   "loss", "trim", "focus", "ema", "trunk", "head"); update the
   docstring diagram. ADDITIVE ONLY: every existing banner line
   byte-identical (pending gate recipes grep them). Do NOT touch
   the "loss_mode" run-line label (deferred, recorded).
2. D-DOC2: the five seven-vs-eight sites (exact lines in the
   note; the YAML edit uses the paste-ready block in the note).
3. D-DOC3: the two module-docstring inventories.
4. D-DOC4/5/6: README edits per the note (EMA everywhere it is
   owed; berhu + the whitened-norm BerHu caveat; the appendix
   function inventories; the section 6 phase/eval-bs/demotion
   sentences; the section 3 rows).
5. D-DOC7: the train_single driver header enumeration.
6. D-DOC8: the three nits.
7. D-DOC9: the TRF MLP token-width pin made explicit at every
   surface listed in the note (user ruling: the pin is the rule;
   do NOT add a width knob — [[trf-mlp-width-knob]] is shelved).
Gates: GDOC-A..E (recipes in the note). Report: raw gate outputs
pasted, every YAML change as a paste-ready block, deviations
declared. Do not commit: leave the diff uncommitted and print the
suggested commit command.
### END

## Status

AUDIT DELIVERED 2026-07-06. Fixes NOT yet implemented; the handoff
above is ready for the user to relay. On the Implementer's report:
Architect re-audit closes GDOC-A..E, then this is one commit unit
(suggested sentence: "Documentation truth pass: banner prints the
resolved loss + ema blocks (consumed-view); eight-key phase
enumerations; README gains the EMA / berhu / validator-layer /
demotion / derived-eval-bs coverage; the TRF MLP token-width pin
explicit (doc audit D-DOC1..9, Architect-verified)").

Addendum 2026-07-06: D-DOC9 added by user ruling — the TRF MLP
interior stays pinned to the token width ("a fair rule"); the
mlp_width knob spec ([[trf-mlp-width-knob]]) is SHELVED and the
explicit-pin documentation joins this handoff instead.

## Resume state (Implementer appends below)

### 2026-07-06 — Implementer (Opus 4.8) execution: D-DOC1..9

Clean base (tip 8bb5484). All nine deltas implemented; gates
GDOC-A..E pass on the Mac. Scope note: the relayed ARCHITECT_HANDOFF
listed D-DOC1..8 only; the note (read per the handoff) had gained
D-DOC9 by addendum after the block was copied, and the Status folds
D-DOC9 into this same commit unit (D-DOC1..9), so I executed it too.
Flagged in the report.

**Done, per delta:**

- D-DOC1 (the only code change): print_design's sub-block tuple
  (experiment.py) gained "loss" and "ema" in spec order
  (optimizer, lr, scheduler, loss, trim, focus, ema, trunk, head);
  the docstring lines-printed diagram updated to match. Additive: the
  loop still prints only blocks present in ta, so every prior line is
  byte-identical and the two new lines insert after scheduler / focus.
  The "loss_mode" run-line label left untouched (deferred, recorded).
- D-DOC2 (five sites -> eight keys + ema): training.py
  validate_phase_block docstring ("other six keys", "eight-key
  whitelist"); training.py:2073 comment; training.py:2280 comment +
  the enumeration gained ema; experiment.py __init__ trunk/head entry
  (lr / scheduler / loss / trim / focus / clip / rewind / ema);
  train_single YAML per-phase comment (the paste-ready block).
- D-DOC3 (module docstrings): training.py gained the config validator
  / derivation-layer paragraph + the loop-summary "berhu-blend / EMA
  annealing, best-epoch tracking on the EMA average" + the five-mode
  list (chi2 / sqrt / sqrt_dchi2 / berhu / berhu_capped, verified
  against the code, NOT mse/huber). loss_functions.py: the transform
  ladder now names berhu / berhu_capped; anneal_value now "four knobs
  (trim, focus, berhu sqrt-blend, EMA horizon)".
- D-DOC4/5/6 (README): section 2 step 7 EMA-selection sentence;
  section 3 loss_functions + training rows; section 6 YAML tour gained
  the ema: block, the eight-key phase whitelist, the single-phase
  demotion sentence, and the derived-eval-bs sentence; section 10
  loss_functions appendix (mode ladder + a dedicated berhu bullet with
  the whitened-norm BerHu naming caveat: delta = sqrt(knot),
  per-sample Mahalanobis aggregate, knot/cap in chi2 units); section
  10 training appendix (validate_* + derive_* bullets, the loop entry
  gained the Polyak clause); section 10 experiment appendix
  (validate_param_cuts / validate_sizes / resolve_phase_args /
  validate_sweep_paths bullets, print_design gained the consumed-view
  clause).
- D-DOC7 (train_single driver header): the phase-override list gained
  scheduler + ema (eight-key whitelist) and a top-level ema key entry.
- D-DOC8 (three nits): tune YAML search-leaves note gained
  ema.horizon_epochs / ema.anneal.*; emulator_designs.py module
  docstring gained the DesignSpec / head_block sentence; experiment.py
  __init__ model_cls now "(ResMLP / ResCNN / ResTRF)".
- D-DOC9 (TRF token-width pin explicit, doc-only, knob shelved): the
  pin clause ("every layer at the token width / dim -> dim, pinned to
  the bin length, no width knob, n_mlp_blocks is depth only") added
  beside n_mlp_blocks at all seven surfaces — TRFBlock docstring +
  construction comment (building_blocks); ResTRF + TemplateResTRF
  head-knob entries; experiment.py __init__ trf entry; README section
  6 trf sentence + section 10 TRFBlock + ResTRF; train_single YAML
  n_mlp_blocks comment; tune YAML n_mlp_blocks comment; train_single
  driver header trf clause. The sweep YAML defers to the train YAML
  ("see the train YAML") so needs no clause.

**Gate evidence (raw, Mac; no torch):**

- GDOC-A (exec-extract print_design's sub-block loop): the live tuple
  is (optimizer, lr, scheduler, loss, trim, focus, ema, trunk, head);
  fed a resolved ta with loss {mode berhu_capped, berhu {knot, cap,
  anneal}} + ema {horizon_epochs, anneal} + all prior blocks -> exactly
  one "loss: {...}" line and one "ema: {...}" line printed; all 7
  pre-fix lines byte-identical and in the same relative order; post =
  pre + exactly the 2 new lines. ALL PASS.
- GDOC-B: tree-wide "seven-key" / "seven keys" / "the other five keys"
  -> zero hits; each of the five D-DOC2 sites lists all eight keys with
  ema present. PASS.
- GDOC-C: both module docstrings name berhu + berhu_capped and all
  four anneal_value consumers; training.py's names the validator layer
  (validate_phase_block / validate_loss / validate_berhu /
  validate_ema + derive_eval_bs / derive_ema_beta). PASS.
- GDOC-D: README greps — EMA present in sections 2/6/10;
  derive_eval_bs present; the anneal_value table row updated; the
  BerHu whitened-norm caveat sentence present; the demotion sentence
  in section 6; the D-DOC9 token-width-pin clause present beside
  n_mlp_blocks at every listed surface. PASS.
- GDOC-E: no non-README addition over 90 cols (README bullets / table
  rows are the file's one-line convention: 90 such lines pre-date this
  diff); zero ` -- `; added all-caps tokens are established acronyms
  only (MLP, EMA, YAML, LN, FFN, CMB — no new emphasis); both touched
  YAML comment blocks keep their hash-column alignment (train_single
  col 28, tune col 30); whole-tree py_compile clean.
- Extra proof (AST-minus-docstrings, HEAD vs working tree over all
  seven touched .py): only experiment.py is CODE CHANGED (the one
  print_design tuple); training.py, loss_functions.py,
  emulator_designs.py, emulator_designs_building_blocks.py,
  IA/emulator_designs.py, and the train_single driver are all
  CODE-IDENTICAL. The entire pass touched exactly one code node.

**Deviations / declared observations (for the re-audit):**

- (scope) D-DOC9 executed from the note's addendum, not the relayed
  handoff (which stopped at D-DOC8); the Status folds it into this
  commit unit. Flag if the intent was a separate unit.
- (out-of-scope enumeration rot, NOT fixed) three partial phase-key
  enumerations sit outside the audit's five D-DOC2 sites and now read
  stale; left untouched to respect scope, surfaced here:
  1. experiment.py train() call comment "(lr / loss / trim / focus)"
     — reads illustrative ("the head may want e.g. ..."), missing
     scheduler / clip / rewind / ema.
  2. training.py run_emulator docstring two-phase diagram legend
     "lr / loss / trim / focus / clip / rewind per phase" (6 keys;
     missing scheduler + ema; scheduler is named on the adjacent
     trunk-phase line).
  3. train_single YAML the detailed full-replacement walkthrough
     "loss (the nested {mode, berhu} block) / clip / rewind replace
     their value" — missing ema; deliberately outside the D-DOC2
     paste-ready block (the Architect scoped that block to end at the
     eight-key summary line, "line 149's continuation follows
     unchanged").
- No behavioral deviation: the one code edit is the additive tuple;
  every existing banner line stays byte-identical (GDOC-A).

Open: none Mac-side (documentation pass; no workstation leg). Awaiting
Architect re-audit of GDOC-A..E, then the single commit unit.

## Architect re-audit 2026-07-06: ACCEPTED, no deltas

Every gate reproduced with my own harnesses (never the Implementer's
outputs):

- GDOC-A: own AST-minus-docstrings harness over all seven .py — six
  CODE-IDENTICAL, experiment.py the only CODE CHANGED; the diff's one
  code hunk is exactly the additive tuple ("loss" after scheduler,
  "ema" after focus; the f-string and the `if block in ta` guard
  untouched), so prior-line byte-identity holds structurally.
- GDOC-B: tree-wide "seven-key" / "seven keys" / "other five keys" ->
  zero; eight-key language verified at training.py 16 / 229-232 / 274
  / 289 / 2010 / 2073 / 2280, experiment.py 620, the YAML block
  (applied verbatim from this note's paste-ready form).
- GDOC-C: both module docstrings read directly — the validator-layer
  paragraph, the four anneal_value consumers, the five-mode ladder.
- GDOC-D: README verified at 303 (Polyak, section 2), 346/348
  (section 3 rows), 462 (demotion, section 6), 802/804 (anneal_value
  + the berhu bullet with the whitened-norm caveat), 835-849
  (validator / derive / resolve bullets); D-DOC9 pin at all seven
  surfaces (the IA TemplateResTRF clause confirmed IN THE DIFF — my
  first grep missed it only because "token width" wraps lines).
- GDOC-E: added non-README lines all <= 90 cols; zero double-hyphen;
  YAML hash columns aligned in the diff; py_compile all seven OK.

Rulings:
- D-DOC9-from-the-addendum: ACCEPTED — the note is the spec of
  record and the handoff says to read it; one commit unit as folded.
- The three surfaced enumerations: CONFIRMED rot (obs 2, the
  run_emulator two-phase diagram legend at training.py:1909, and
  obs 3, the YAML full-replacement walkthrough at line ~155, are
  plain enumeration rot missing scheduler/ema resp. ema; obs 1, the
  experiment.py:1375 train() comment, is borderline-illustrative but
  worth completing). Recorded as D-DOC2b, assigned to the NEXT cycle
  (freeze_trunk unit A rewrites the run_emulator two-phase diagram
  and the YAML schedule walkthrough anyway; head-activation unit B
  touches the train() comment region) — no extra round-trip for this
  green unit. The Implementer's scope discipline (surface, don't
  creep) was correct.

COMMIT-READY as ONE unit. Suggested sentence: "Documentation truth
pass: banner prints the resolved loss + ema blocks (consumed-view);
eight-key phase enumerations; README gains the EMA / berhu /
validator-layer / demotion / derived-eval-bs coverage; the TRF MLP
token-width pin explicit (doc audit D-DOC1..9, Architect-verified)".
