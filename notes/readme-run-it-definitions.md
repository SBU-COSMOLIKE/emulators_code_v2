---
name: readme-run-it-definitions
description: "SPEC 2026-07-07 (Architect): restructure the README's Run it section — user critique: 'this is the first section and you dont define things. What is the N_train learning curve? What do you mean across all GPUs? what is activation bake-off? what is Optuna?... break this one block into several so you can add text in between... not overly verbose... cant be overly scary.' The five-command bash block splits into a setup block + five per-driver blocks, each preceded by 1-3 definition sentences (prose given VERBATIM in this note: the learning-curve falling-vs-flat-tail reading, the one-training-per-GPU parallelism sentence pointing at the Multi-GPU subsection, the Optuna two-liner, the bake-off head-to-head reading). Commands stay byte-identical (only split); ~12 added prose lines. RIDES the weight-decay unit (the next doc-carrying unit after GAN). Gate GRI-A. NOT implemented."
metadata:
  node_type: memory
  type: project
---

# Run it: break the block, define the terms (spec)

User critique 2026-07-07 (verbatim intent): the first section dumps
five commands with undefined jargon; split into several blocks with
brief definitions between — didactic, not scary, not verbose.

The prose, VERBATIM (the Architect's chat draft, user-seen):

- Intro (GENERALIZED, user directive 2026-07-07 — "you assume
  everyone will run on my workstation... this code at some point
  will be for wide use"): "Training needs a machine with a working
  Cocoa installation — cosmolike supplies the data-vector mask and
  covariance — and, in practice, a CUDA GPU; the emulator/ package
  itself is pure PyTorch and can be read or developed anywhere.
  Every driver reads the same three path flags, so set the driver
  folder once:" + the setup block (the existing comment lines + the
  D= line only).
- train_single: "One training run — train the YAML's model once;
  --diagnostic adds a multipage PDF of accuracy diagnostics:"
- sweep_ntrain: "The N_train learning curve — how does accuracy
  improve as the training set grows? This driver retrains the same
  model at several training-set sizes and plots the error against
  N: a curve still falling at the largest N says more data will
  help; a flat tail says the model, not the data, is the limit.
  Sweep points are independent trainings, so they run in parallel —
  one whole training per GPU, all visible GPUs by default
  ([Multi-GPU](#multi-gpu) below):"
- sweep_hyperparam: "A one-knob sweep — one full training per value
  of a single YAML-chosen knob (learning rate, kernel size, batch
  size, ...), to see that knob's effect in isolation; the knob and
  values live in the [sweep: block](#sweep-block):"
- tune_single: "A hyperparameter search — Optuna
  (https://optuna.org) is a search library: it proposes trial
  settings, watches the results, and concentrates new trials where
  the metric improves. The searched ranges are the YAML's
  [default, min, max, kind] leaves; --n-trials bounds the study:"
- bakeoff: "The activation bake-off — trains the same model once
  per activation family over a grid of training sizes and overlays
  their learning curves: a head-to-head showing whether a family
  genuinely learns faster (a lower curve everywhere) or just ties:"
- The --gpu-pack paragraph after the five blocks stays as is.

Rules: the five commands byte-identical to today (only the block is
split; the setup comment lines move into the setup block); bold
lead-ins as shown; the two in-file anchors (multi-gpu, sweep-block)
resolve; budget ~12 added prose lines, nothing else reworded.

ALSO (same directive): the workstation-assumption sweep. Six
user-facing sites say "the workstation" (grep 2026-07-07): the
README intro (above), emulator/README.md:70, and four driver
headers (train_single:33, sweep_ntrain:82, sweep_hyperparam:83,
bakeoff:74). Replace each with the general form — "a machine with
a working Cocoa installation (cosmolike)" / for the code map:
"...imports cosmolike, so training runs on a machine with a
working Cocoa installation; the library everywhere else is pure
PyTorch." The driver-header edits are comment/docstring lines only
(AST code-identity must hold). notes/ is EXEMPT (internal knowledge
base; it references the group's real machines).

Gate GRI-A: commands byte-identical (diff shows only splits +
prose); the four defined terms present (learning curve, per-GPU
parallelism, Optuna, bake-off); anchors resolve; grep -i
workstation over README.md + emulator/README.md + the five drivers
returns ZERO; AST-minus-docstrings identical on touched .py;
doc-only; scans.

Packaging: RIDES the weight-decay unit (the next doc-carrying unit
after GAN lands) as a second README item — or its own micro-unit if
the user prefers it sooner. NOT implemented.

## The full didactic read (2026-07-07; same unit — this note is now
## "the didactic pass")

User: "give an entire read of the readme. Look for places where
things are not defined, not explained... provide some minimal
context." Architect findings, verified by first-use line numbers:

SYSTEMIC: the reorg inverted the vocabulary dependency — whitening,
theta order, staging, and the goal metric are DEFINED in the
pipeline appendix (section 12, line ~701) but USED from the intro
(line 8) and throughout the YAML chapter (whitened residual norm at
loss ~302; full-whitened / theta order / whitened templates at
model ~470-481; "the frac>0.2 goal" at ~301 explained only at
~914). FIX DEVICE: a compact vocabulary box closing section 2 (The
YAML file) — six one-liners, each with its pointer:

    Five terms the chapter uses (details: appendices 12-13):
    - data vector (dv): the masked cosmic-shear two-point functions
      xi+/- stacked into one vector — what the network predicts.
    - chi2: prediction error measured in the analysis covariance,
      r^T Cinv r (appendix 13). The headline metric, written
      frac>0.2 in the logs: the fraction of validation cosmologies
      with delta-chi2 above 0.2 — the goal is to drive it down.
    - whitened: rotated and rescaled so the components are
      decorrelated with unit variance — the form the network sees,
      input and output (appendix 12).
    - theta order: the data vector re-sorted to vary smoothly along
      the angular axis — the basis the correction heads work in.
    - trunk / head: every architecture is a shared ResMLP trunk;
      rescnn / restrf add a gated correction head on top (section 10).
    - dump: the big on-disk (params, dv) table the physics code
      wrote; training memmaps it (reads slices from disk, never the
      whole file) and stages only the rows it needs (section 3).

LOCAL FINDINGS (one-liners / pointers, exact spots):

1. The intro (lines 4-9) is jargon-dense (masked / xi / 3x2pt /
   cosmolike undefined before the one-line flow). Fix: 3-4 plain
   sentences BEFORE the flow line: an emulator replaces the
   expensive physics pipeline inside inference loops; xi = the
   cosmic-shear two-point correlation functions; cosmolike (inside
   Cocoa) supplies the analysis mask + covariance; accuracy is
   judged as chi2 — error in the units inference cares about.
2. Section 1: already covered by this unit (split + definitions +
   workstation sweep). Additions: the sweep_ntrain prose names the
   metric ("the error metric: frac>0.2, defined in section 2");
   "chains" gets the clause "(the project's data folder,
   --root/chains)".
3. Section 2: the vocab box above; "[default, min, max, kind]"
   gains "(kind = int | float | log)".
4. Section 3: "dump" + "memmap" covered by the box; keep one
   parenthetical at first use in the section.
5. Section 4: rewind mentions the plateau lr cut + patience before
   the scheduler exists -> "(the scheduler's lr cut, section 6)".
   "monster-outlier batch" -> "(a batch with one extreme sample)".
6. Section 5: "trimmed, focally weighted mean" gains "(sections
   7-8)"; "whitened residual norm" -> the box covers it.
7. Section 6: the sqrt rule gains its why: "(bigger batches average
   away gradient noise, so the step can grow with sqrt(bs))";
   "fused on CUDA" -> "(a faster fused kernel)".
8. Section 9: "Polyak weight average" gains "(a running average of
   the network weights; the averaged copy is what ships)".
9. Section 10 (the jargon hotspot): ONE new sentence opening the ia
   paragraph: "Intrinsic alignments (IA): galaxies have correlated
   intrinsic shapes independent of lensing — the main astrophysical
   contaminant of cosmic shear; the analysis models it with
   amplitude parameters." Then: "(xi+/-, source-pair) bins" ->
   "(one bin = one source-redshift-bin pair of xi+ or xi-)";
   "gate" -> "(a learnable scalar scaling the correction, starting
   near 0)".
10. Metric spelling unified: the appendix-12 "f(dchi2 > 0.2)" gains
    "(the frac>0.2 of the logs)" so the three spellings meet once.

Budget: the box ~14 lines + ~12 one-liner lines total. Not scary;
nothing else reworded. Gate GRI-B: the box present with its six
entries + pointers; each numbered fix present at its spot; the
metric spellings tied; anchors resolve; doc-only; scans.

PACKAGING (updated): this note is now the umbrella for ONE doc-only
unit, "the README didactic pass" = the Run-it split/definitions +
the workstation sweep + the vocab box + the ten local fixes. Rides
after GAN lands (before or beside the weight-decay unit — the
weight-decay README treatment stays with its code fix).

## Handoff

### ARCHITECT_HANDOFF
Task: the README didactic pass (spec: THIS note in full —
notes/readme-run-it-definitions.md: the Run-it split with the
verbatim definition prose, the workstation-assumption sweep over
the six user-facing sites, the section-2 vocabulary box, and the
ten local one-liners). Base: the GAN commit (relu/tanh +
model.norm); `git log -1` must show it — else STOP. DOC-ONLY
except the four driver-header workstation lines (comment/docstring
only; AST code-identity must hold). Rules: the five Run-it
commands byte-identical; prose verbatim from this note; budget
respected; notes/ exempt from the workstation sweep. Gates GRI-A +
GRI-B (recipes in this note). Report: IMPLEMENTER_HANDOFF + resume
state appended here, raw gate outputs, deviations declared. Do not
commit: print the suggested commit command.
### END

## Implementer resume state (2026-07-07, Opus, UNIT 1 of the overnight queue, base 8ad25a1)

IMPLEMENTED, uncommitted (rides UNIT 2 into one combined commit). Doc-only
except four driver-header comment lines (AST-identical).

- Run it (section 1): the intro is generalized (workstation site #1 gone —
  "a machine with a working Cocoa installation ... in practice a CUDA GPU;
  the emulator/ package is pure PyTorch") + the five-command block split into
  a setup block + five per-driver blocks, each preceded by the verbatim
  definition prose (learning-curve falling-vs-flat reading, the per-GPU
  parallelism sentence -> [Multi-GPU], the Optuna two-liner, the bake-off
  head-to-head). Commands byte-identical (GRI-A). Fix #2 folded in (the
  sweep_ntrain metric names frac>0.2 -> section 2; the chains gloss).
- Workstation sweep: all six sites cleared (README intro, emulator/README.md,
  train_single / sweep_ntrain / sweep_hyperparam / bakeoff headers). tune has
  none. grep -i workstation over the six user-facing files = 0. The four
  driver edits are comment lines only -> AST identical vs HEAD (verified).
- Section 2 vocabulary box: six one-liners (dv, chi2, whitened, theta order,
  trunk/head, dump) with resolving pointers to appendices 12-13 / sections
  3, 10. Fix #3 (kind = int | float | log) folded into section 2.
- The ten local fixes: 1 (intro plain sentences), 2 (above), 3 (above), 5
  (rewind -> scheduler/section-6 pointer; monster-outlier gloss), 6 (trimmed
  mean -> sections 7-8), 7 (lr sqrt why + fused gloss; section 6, disjoint
  spans from UNIT 2), 8 (Polyak gloss), 9a (the IA sentence), 9b (gate gloss),
  10 (f(dchi2>0.2) tied to frac>0.2). #4 (dump/memmap) is satisfied by the box
  + section 3's existing memmap parenthetical (no new edit).

Gates GRI-A + GRI-B: ALL PASS (commands byte-identical; four terms defined;
workstation grep = 0; four drivers AST-identical; box present with six entries
+ resolving pointers; each numbered fix present; metric spellings tied; anchors
resolve; doc-only; py_compile clean).

Deviations declared:
1. Fix #9c's target phrase "(xi+/-, source-pair) bins" is NOT in section 10 —
   the earlier README reorg relocated the detailed ResTRF description to the
   code map (emulator/README.md). Applied the gloss there ("one bin is one
   source-redshift-bin pair of xi+ or xi-"), honoring the note's intent;
   declared as a spec-vs-tree location mismatch.
2. The vocab box header: the note wrote "Five terms" but lists SIX bullets
   (GRI-B asserts six). Rendered "Six terms" to match the count.
3. Verbatim prose gained house-style backticks on code tokens (N_train,
   --diagnostic, sweep:, frac>0.2, [default, min, max, kind]) — text
   unchanged, matching the README's code-span convention.
4. Section 6 overlaps UNIT 2 (see the UNIT 2 note): the Architect's "disjoint
   README regions" is section-level optimistic; fix #7 (lr why + fused gloss)
   and UNIT 2 (weight_decay clause + treatment) edit NON-overlapping spans of
   the optimizer bullet, so both landed cleanly — no collision-STOP needed.

Awaiting Architect re-audit.
