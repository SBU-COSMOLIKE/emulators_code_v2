---
name: compute-data-vectors-import
description: "SPEC 2026-07-07 (Architect): import the training-set generator + document its logic. (1) NEW folder compute_data_vectors/ holding dataset_generator_lensing.py — imported VERBATIM from the user's production copy (byte-identical except the enumerated comment-path lines: emultrf/emultraining -> emultrfv2/compute_data_vectors; it is battle-tested MPI code, NO house-style retrofit, the style rules bind new code only). (2) A NEW main-README appendix 'Generating the training set' teaching the sampling logic: the tempered posterior log p_T = [-(1/2)(theta-theta_0)^T Sigma-tilde^{-1} (theta-theta_0) + log pi(theta)]/T (theta_0 = fiducial; Sigma-tilde = the Fisher covmat with correlations CLIPPED to maxcorr; T = temperature widening the cloud beyond the posterior — the _cs_<T> file tag the trainer parses), emcee sampling + dedupe + thin, the --unif alternative, --boundary < 1 shrinking val/test inside the training support, the failure protocol (failed rows zeroed + flagged; --loadchk recomputes), and the OUTPUT CONTRACT that closes every loop: .1.txt columns = weights/lnp/params/chi2 (exactly the trainer's staging slice), .paramnames = cobaya names (the naming loop), .covmat = the trainer's train_covmat (input whitening!), .ranges, the .npy dvs. Disambiguation: the generator YAML's train_args block (probe/ord/fiducial/params_covmat_file) is UNRELATED to the trainer's train_args — two stages, two schemas, named loudly. Placement: new appendix before AI-Usage (which stays last). Gates GDG-A (verbatim-import diff), GDG-B (docs + anchors), GDG-C light (py_compile only — compile, not import; mpi4py/emcee absent on the Mac; generation runs are science ops, not gates). Sequenced after save-schema-v2 (README collision avoidance), independent of the cobaya adapter. IMPLEMENTED 2026-07-07 (Opus, base b2afca6), uncommitted; GDG-A/B/C PASS (verbatim import = only the line-20 header path; appendix = README section 17, AI-Usage -> 18; the two carried save-v2 docs landed)."
metadata:
  node_type: memory
  type: project
---

# compute_data_vectors/: the generator import + its README appendix

User directive 2026-07-07: "the readme needs an appendix with this
logic -> how the parameters are set from training. On that, we need
a compute_data_vectors folder where we need to add the Python
script to compute the data vector." The cosmic-shear generator
(dataset_generator_lensing.py, ~1120 lines, MPI + emcee + cobaya)
was provided and read in full.

## Part 1: the folder + the verbatim import

- NEW compute_data_vectors/ at the repo root (beside emulator/,
  cobaya_theory/). File name KEPT: dataset_generator_lensing.py
  (the trainer YAML's data-block comment already references it).
- VERBATIM IMPORT: byte-identical to the user's production copy
  EXCEPT the comment-only path examples in the header
  (emultrf/emultraining -> emultrfv2/compute_data_vectors). It is
  battle-tested MPI production code — NO house-style retrofit, no
  reformat, no rename (the house rules bind NEW code; a style pass
  on 1100 working lines is churn risk with zero payoff). The
  README appendix, not the script, carries the didactic layer.
- Future probes (ggl / gc per its probe whitelist) inherit the
  folder naming.

## Part 2: the README appendix — "Generating the training set"

Placement: a new numbered appendix in the main README AFTER the
precedence appendix, BEFORE AI-Usage (which stays last, the
standing rule). Section-3 (data) gains one pointer sentence ("where
the dumps come from: appendix N"). Content (didactic, the chapter
style — equations + small blocks, not verbose):

1. The goal in two sentences: training density should cover WHERE
   CHAINS WILL EXPLORE, broader than the posterior itself — an
   emulator is only trustworthy inside its training cloud.
2. THE TWO SAMPLING MODES, presented as peers (user directive
   2026-07-07 — the uniform option gets equal billing, not a
   clause):
   - Gaussian / tempered (--unif 0, the default): the tempered
     posterior below — the training cloud shaped like a widened
     posterior.
   - Uniform (--unif 1): uniform draws inside the
     (temperature-stretched) hard bounds — a flat cloud filling the
     whole box, no posterior shaping; lnp is set to 1 (no weights).
     THE SUBTLETY the script header records and the appendix must
     keep: even uniform sampling NEEDS --temp, because T sets the
     hard-boundary stretch for parameters whose priors are Gaussian
     / unbounded (bounds widened by T x width / 5) — without it
     those parameters have no box to be uniform in. Output files
     tag as _<probe>_unifs instead of _<probe>_<T> (the trainer's
     data: filenames differ accordingly).
   When to use which, one line: tempered = production training sets
   (density where chains explore); uniform = coverage studies /
   stress tests of the emulator far from the posterior.
3. The tempered posterior (the heart of --unif 0):
   $$\log p_T(\theta) = \frac{1}{T}\left[
   -\tfrac{1}{2}(\theta-\theta_0)^\top \tilde\Sigma^{-1}
   (\theta-\theta_0) + \log \pi(\theta)\right]$$
   legend: theta_0 = the fiducial; T = the temperature (--temp; the
   _cs_<T> tag in every output filename, the same tag the trainer's
   file names and the drivers' t<T> run tag parse); pi = the cobaya
   prior (hard bounds respected; infinite-prior bounds stretched by
   T x width / 5); Sigma-tilde = the Fisher/params covmat with
   correlations CLIPPED to |corr| <= maxcorr (default 0.15).
4. WHY each knob, one line each: T flattens the likelihood so the
   cloud extends past the posterior (T = 1 would hug it; chains at
   the posterior edge would leave the training support); maxcorr
   fills the volume PERPENDICULAR to degeneracy directions (a raw
   Fisher covmat is a thin pancake along the degeneracy — the
   emulator needs volume, not a line); --boundary < 1 shrinks
   val / test INSIDE the training support (accuracy degrades at the
   cloud's edge, so validation must not sit on it).
5. The machinery, one short paragraph: emcee (DE + snooker moves)
   samples log p_T (tempered mode), dedupe + thin to --nparams, tau
   diagnostics printed; MPI master/worker computes the dvs via the cobaya model
   (CAMB/cosmolike per the generator YAML), checkpointed
   (--freqchk / --loadchk / --append), failures zeroed + flagged in
   the failfile and recomputable by rerunning with --loadchk 1.
6. THE OUTPUT CONTRACT (a small table — this is where every loop
   closes): <paramfile>_cs_<T>.1.txt (uniform runs: _cs_unifs) (columns weights / lnp /
   params / chi2* — exactly the slice the trainer's staging drops),
   .paramnames (FIRST COLUMN = the cobaya names -> ParamGeometry ->
   the h5 -> get_requirements: the naming loop), .covmat (= the
   trainer's data.train_covmat — the INPUT WHITENING basis),
   .ranges, <datavsfile>_cs_<T>.npy (the dv dump the trainer
   memmaps), <failfile>_cs_<T>.txt.
7. DISAMBIGUATION, loud: the generator YAML is a COBAYA yaml
   carrying its OWN train_args block (probe, ord, fiducial,
   params_covmat_file) — UNRELATED to the emulator trainer's
   train_args. Two stages, two schemas; the appendix names both.
8. A run-it block (the mpirun example, emultrfv2 paths, lsst_y1
   flavor to match the README convention; the script header keeps
   the user's roman_real example verbatim).

## Part 3: the two prose docs CARRIED from save-schema-v2

The save unit ([[save-schema-resolved-config]]) landed as b2afca6
with two documentation items explicitly carried here ("GDG-B gains
them"):

- train_single_emulator_cosmic_shear.py driver header: a --save
  paragraph — what --save emits (the .emul/.h5 pair; the h5 is
  schema v2: the fully-resolved consumed config + the model recipe;
  rebuild_emulator reconstructs from the file alone).
- Main README, the Run-it outputs sentence gains: "the h5 carries
  the fully-resolved config (schema v2) — a saved emulator rebuilds
  bit-exactly even if code defaults later change."

## Gates

- GDG-A (verbatim import): diff of compute_data_vectors/
  dataset_generator_lensing.py against the user-provided copy shows
  ONLY the enumerated header path lines; python -m py_compile
  passes (compile only — mpi4py / emcee / cobaya need not be
  importable on the Mac). The __pycache__/ beside it is never
  staged.
- GDG-B (docs): the appendix present with the tempered-posterior
  equation (GRO-G-safe symbols + legend), the TWO sampling modes as
  peers (incl. the uniform-still-needs---temp subtlety + the _unifs
  tag), the why-lines, the output-contract table, the
  two-train_args disambiguation, the section-3 pointer; PLUS the
  two carried docs of Part 3; anchors resolve; AI-Usage still last;
  the five-rule math scanner passes.
- GDG-C (light): no generation-run gate — the script is the user's
  proven production tool; full runs are science operations, not
  unit gates.

## Handoff (base landed; ready to relay; independent of cobaya)

### ARCHITECT_HANDOFF
Task: the compute_data_vectors import + the training-set appendix
(spec: notes/compute-data-vectors-import.md in full; the VERBATIM
import rule is binding — byte-identical except the enumerated
header path lines, no style retrofit). Base: commit b2afca6 ("Save
schema v2: ..."); `git log -1` must show it — else STOP. The script
is ALREADY in the tree, untracked, at
compute_data_vectors/dataset_generator_lensing.py — that copy IS
the source of truth; do not regenerate it, only apply the
enumerated header-path edits and verify GDG-A against `git diff` /
the header. Never stage compute_data_vectors/__pycache__/. Scope:
parts 1 + 2 + 3 exactly (part 3 = the two prose docs carried from
save-schema-v2). Gates GDG-A/B/C. Report: IMPLEMENTER_HANDOFF +
resume state appended here, raw gate outputs, deviations declared.
Do not commit: print the suggested commit command with explicit
paths.
### END

## Status

IMPLEMENTED 2026-07-07 (Opus, base b2afca6), uncommitted,
COMMIT-READY. Parts 1 + 2 + 3 all landed; GDG-A/B/C PASS.
Independent of the cobaya adapter (either order). Suggested commit
sentence: "Training-set generator imported verbatim under
compute_data_vectors/ + the README appendix on the
tempered-sampling logic (two sampling modes, T, maxcorr, boundary,
the output contract that feeds the trainer) + the carried save-v2
prose docs (gates GDG-A/B Architect-verified)".

## Implementer resume state (2026-07-07, Opus, base b2afca6) — DONE

- PART 1 (verbatim import): the untracked
  compute_data_vectors/dataset_generator_lensing.py got exactly ONE
  edit — line 20's header path emultrf/emultraining ->
  emultrfv2/compute_data_vectors. GDG-A: diff vs a pre-edit snapshot
  = that single hunk, zero residual emultrf, py_compile clean. The
  roman_real example in the header stays verbatim. __pycache__ is
  never staged (the commit adds the .py file by name, not the dir).
- PART 2 (README appendix): NEW `## 17. Appendix: Generating the
  training set` inserted before AI-Usage (renumbered to `## 18`);
  TOC + the section-3 pointer + all internal anchors updated.
  Content: the goal, the two sampling modes as peers (uniform still
  needs --temp; lnp=1; the _unifs tag), the tempered-posterior
  equation (GRO-G-safe, one line, no backslash+punct), the why-lines
  (temp/maxcorr/boundary), the machinery (DEMove 0.9 / DESnookerMove
  0.1, MPI, checkpointing), the output-contract table (six products),
  the two-train_args disambiguation, an emultrfv2 mpirun block.
  Every quantitative claim was verified against the script:
  y = x - fiducial (l.517), logp = [-0.5 y^T C^-1 y + logprior]/temp
  (l.525), bound stretch temp*width/5 (l.333/335), maxcorr default
  0.15 (l.231), the _unifs vs _{probe}_{temp} tag branch (l.382-389),
  columns weights/lnp/<params>/chi2* (l.633/675).
- PART 3 (carried docs): train_single header --save paragraph now
  names schema v2 + the model recipe + rebuild_emulator (AST
  vs HEAD = IDENTICAL, comment-only); the README Run-it block gained
  the .emul/.h5 + schema-v2 + rebuild + bit-exact outputs sentence.
- GATES: GDG-A (verbatim diff + compile) PASS; GDG-B (docs+anchors+
  five-rule math scanner, all internal anchors resolve incl. the
  renumbered AI-Usage) PASS; GDG-C light (py_compile) PASS.
- DEVIATION / side-note (out of scope, flagged not fixed): the
  AI-Usage prose still reads "under the `dev` folder" — a stale
  reference to the retired /dev install path (RIDER 1 swept the
  emultrf/dev *paths* last unit but not this prose phrasing). A
  one-word doc fix for a future pass; not enumerated here.
- COMMIT: staged by explicit paths (README.md,
  train_single_emulator_cosmic_shear.py,
  compute_data_vectors/dataset_generator_lensing.py, notes/
  compute-data-vectors-import.md, notes/MEMORY.md); NOT
  __pycache__, NOT notes/session-status-2026-07-07b.md.

## Architect audit verdict (2026-07-07, Fable, independent probes)

ACCEPTED — COMMIT-READY. Every check re-run from raw evidence:

- GDG-A PASS, with an independent verbatim proof the handoff didn't
  have: the byte arithmetic. I staged the original at 53,058 bytes;
  the path swap (emultrf/emultraining, 20 chars -> emultrfv2/
  compute_data_vectors, 30 chars) adds exactly +10; the file now
  measures 53,068. Line count 1121 unchanged; zero residual
  emultrf/emultraining; line 20 carries the new path; roman_real
  header example intact; py_compile clean.
- GDG-B PASS. Appendix content = all eight spec items; every
  quantitative claim re-verified against the source myself (logp/temp
  l.525, temp*width/5 l.333-335, maxcorr 0.15 l.231, _unifs branch
  l.382-389, DEMove 0.9 / DESnookerMove 0.1 l.553-554, the
  weights/lnp/params/chi2* header l.633/675, lnp=ones l.587). The
  five-rule math scanner: CLEAN on both READMEs. Anchors: ALL
  resolve — after I fixed TWO bugs in MY OWN checker (it slugged
  headings from code-blanked lines, and it never collected the
  <a name=...> HTML anchors); both were harness-side false flags,
  zero doc failures. The owned pattern held again: calibrate the
  harness on a known-bad file first. README diff = exactly the four
  intended hunks (TOC, Run-it outputs paragraph, section-3 pointer,
  appendix + AI-Usage renumber).
- PART 3 PASS: train_single AST-identical vs HEAD (comment-only)
  with the schema-v2 --save paragraph; the Run-it outputs paragraph
  names .emul/.h5, schema v2, bit-exact rebuild, rebuild_emulator.
- GDG-C PASS (both py_compile).
- The flagged deviation (AI-Usage says "under the `dev` folder") is
  NOT a doc bug to queue: that sentence is the USER'S VERBATIM text
  (the standing AI-Usage rule). Only the user can amend it —
  surfaced to the user as a one-word decision, not fixed.

Suggested commit (user runs it; explicit paths, __pycache__ and the
session-status note stay out):

    git add README.md train_single_emulator_cosmic_shear.py \
      compute_data_vectors/dataset_generator_lensing.py \
      notes/compute-data-vectors-import.md notes/MEMORY.md
    git commit -m "Training-set generator imported verbatim under
    compute_data_vectors/ + the README appendix on the
    tempered-sampling logic (two sampling modes, T, maxcorr,
    boundary, the output contract that feeds the trainer) + the
    carried save-v2 prose docs (gates GDG-A/B Architect-verified)"

After this lands, the queue holds ONE unit: the cobaya Theory
adapter ([[cobaya-theory-adapter]]) — then the generate -> train ->
save -> sample arc is complete.
