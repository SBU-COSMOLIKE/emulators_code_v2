---
name: compute-data-vectors-import
description: "SPEC 2026-07-07 (Architect): import the training-set generator + document its logic. (1) NEW folder compute_data_vectors/ holding dataset_generator_lensing.py — imported VERBATIM from the user's production copy (byte-identical except the enumerated comment-path lines: emultrf/emultraining -> emultrfv2/compute_data_vectors; it is battle-tested MPI code, NO house-style retrofit, the style rules bind new code only). (2) A NEW main-README appendix 'Generating the training set' teaching the sampling logic: the tempered posterior log p_T = [-(1/2)(theta-theta_0)^T Sigma-tilde^{-1} (theta-theta_0) + log pi(theta)]/T (theta_0 = fiducial; Sigma-tilde = the Fisher covmat with correlations CLIPPED to maxcorr; T = temperature widening the cloud beyond the posterior — the _cs_<T> file tag the trainer parses), emcee sampling + dedupe + thin, the --unif alternative, --boundary < 1 shrinking val/test inside the training support, the failure protocol (failed rows zeroed + flagged; --loadchk recomputes), and the OUTPUT CONTRACT that closes every loop: .1.txt columns = weights/lnp/params/chi2 (exactly the trainer's staging slice), .paramnames = cobaya names (the naming loop), .covmat = the trainer's train_covmat (input whitening!), .ranges, the .npy dvs. Disambiguation: the generator YAML's train_args block (probe/ord/fiducial/params_covmat_file) is UNRELATED to the trainer's train_args — two stages, two schemas, named loudly. Placement: new appendix before AI-Usage (which stays last). Gates GDG-A (verbatim-import diff), GDG-B (docs + anchors), GDG-C light (py_compile only — compile, not import; mpi4py/emcee absent on the Mac; generation runs are science ops, not gates). Sequenced after save-schema-v2 (README collision avoidance), independent of the cobaya adapter. NOT implemented."
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
2. The tempered posterior (the heart):
   $$\log p_T(\theta) = \frac{1}{T}\left[
   -\tfrac{1}{2}(\theta-\theta_0)^\top \tilde\Sigma^{-1}
   (\theta-\theta_0) + \log \pi(\theta)\right]$$
   legend: theta_0 = the fiducial; T = the temperature (--temp; the
   _cs_<T> tag in every output filename, the same tag the trainer's
   file names and the drivers' t<T> run tag parse); pi = the cobaya
   prior (hard bounds respected; infinite-prior bounds stretched by
   T x width / 5); Sigma-tilde = the Fisher/params covmat with
   correlations CLIPPED to |corr| <= maxcorr (default 0.15).
3. WHY each knob, one line each: T flattens the likelihood so the
   cloud extends past the posterior (T = 1 would hug it; chains at
   the posterior edge would leave the training support); maxcorr
   fills the volume PERPENDICULAR to degeneracy directions (a raw
   Fisher covmat is a thin pancake along the degeneracy — the
   emulator needs volume, not a line); --boundary < 1 shrinks
   val / test INSIDE the training support (accuracy degrades at the
   cloud's edge, so validation must not sit on it).
4. The machinery, one short paragraph: emcee (DE + snooker moves)
   samples log p_T, dedupe + thin to --nparams, tau diagnostics
   printed; --unif 1 = uniform in the stretched bounds instead;
   MPI master/worker computes the dvs via the cobaya model
   (CAMB/cosmolike per the generator YAML), checkpointed
   (--freqchk / --loadchk / --append), failures zeroed + flagged in
   the failfile and recomputable by rerunning with --loadchk 1.
5. THE OUTPUT CONTRACT (a small table — this is where every loop
   closes): <paramfile>_cs_<T>.1.txt (columns weights / lnp /
   params / chi2* — exactly the slice the trainer's staging drops),
   .paramnames (FIRST COLUMN = the cobaya names -> ParamGeometry ->
   the h5 -> get_requirements: the naming loop), .covmat (= the
   trainer's data.train_covmat — the INPUT WHITENING basis),
   .ranges, <datavsfile>_cs_<T>.npy (the dv dump the trainer
   memmaps), <failfile>_cs_<T>.txt.
6. DISAMBIGUATION, loud: the generator YAML is a COBAYA yaml
   carrying its OWN train_args block (probe, ord, fiducial,
   params_covmat_file) — UNRELATED to the emulator trainer's
   train_args. Two stages, two schemas; the appendix names both.
7. A run-it block (the mpirun example, emultrfv2 paths, lsst_y1
   flavor to match the README convention; the script header keeps
   the user's roman_real example verbatim).

## Gates

- GDG-A (verbatim import): diff of compute_data_vectors/
  dataset_generator_lensing.py against the user-provided copy shows
  ONLY the enumerated header path lines; python -m py_compile
  passes (compile only — mpi4py / emcee / cobaya need not be
  importable on the Mac).
- GDG-B (docs): the appendix present with the tempered-posterior
  equation (GRO-G-safe symbols + legend), the why-lines, the
  output-contract table, the two-train_args disambiguation, the
  section-3 pointer; anchors resolve; AI-Usage still last; scans.
- GDG-C (light): no generation-run gate — the script is the user's
  proven production tool; full runs are science operations, not
  unit gates.

## Handoff (relay after save-schema-v2 lands; independent of cobaya)

### ARCHITECT_HANDOFF
Task: the compute_data_vectors import + the training-set appendix
(spec: notes/compute-data-vectors-import.md in full; the VERBATIM
import rule is binding — byte-identical except the enumerated
header path lines, no style retrofit). Base: the save-schema-v2
commit; `git log -1` must show it — else STOP. The user-provided
script is at the path recorded in the session (ask the user to
place it at compute_data_vectors/ or paste it; verify against
GDG-A either way). Scope: parts 1 + 2 exactly. Gates GDG-A/B/C.
Report: IMPLEMENTER_HANDOFF + resume state appended here, raw gate
outputs, deviations declared. Do not commit: print the suggested
commit command.
### END

## Status

SPEC DELIVERED 2026-07-07, NOT implemented. Sequenced after
save-schema-v2 (README collisions), independent of the cobaya
adapter (either order). Suggested commit sentence: "Training-set
generator imported verbatim under compute_data_vectors/ + the
README appendix on the tempered-sampling logic (T, maxcorr,
boundary, the output contract that feeds the trainer; gates
GDG-A/B Architect-verified)".
