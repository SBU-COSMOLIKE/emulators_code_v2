---
name: session-status-2026-07-07b
description: "Architect session state at the 2026-07-07 evening compaction (compaction 4). NINE commits landed since 8bb5484, every one Architect-audited: 2d2f68d doc-audit D-DOC1..9 -> ebd9869 freeze_trunk -> a95f8df per-head activation -> ddb6c98 precedence appendix -> fc4655a the YAML chapter -> 29b23dd the two-README reader-first split (+ parallel/ deleted) -> 83a1e58 model-section polish (TATT + subsection blocks + n_heads diagram) -> 8ad25a1 relu/tanh + model.norm -> 75c429e overnight (didactic pass + weight-decay allowlist) -> 378dc95 NPCE YAML wiring. IN FLIGHT (uncommitted in the tree): save-schema-v2 CORE + riders 1-5, ACCEPTED WITH TWO DELTAS awaiting the Implementer's micro-fix (D-SV1 compile_mode shared constant; D-SV2 the eval_bs stash) — then ONE combined commit (sentence in the note). STAGED untracked: compute_data_vectors/dataset_generator_lensing.py (verbatim, for its own unit). QUEUE after the save commit: the generator import + appendix, the cobaya adapter (either order; both specs handoff-ready). Read this note first; per-unit detail lives in each spec note."
metadata:
  node_type: memory
  type: project
---

# Session status, 2026-07-07 evening compaction (Architect)

Branch claude/amazing-keller-e798b6; the user merges to main and
commits everything (standing rule). Read [[MEMORY]] top-down; this
note is the resume point; each unit's full detail is in its own
spec note.

## The commit chain since 8bb5484 (all Architect-audited)

    2d2f68d  doc-audit D-DOC1..9 ([[audit-docs-2026-07-06]])
    ebd9869  freeze_trunk: joint phase 2 ([[freeze-trunk-joint-phase2]])
    a95f8df  per-head activation + alias + license
             ([[head-activation-per-component]])
    ddb6c98  the precedence appendix ([[readme-precedence-appendix]])
    fc4655a  the README YAML chapter ([[readme-yaml-chapter]])
    29b23dd  the reader-first two-README split + parallel/ DELETED
             ([[readme-reorg-two-readmes]])
    83a1e58  model-section polish: TATT polynomial + mlp/activation/
             cnn code blocks + the n_heads diagram (addenda in
             [[readme-yaml-chapter]])
    8ad25a1  relu + tanh + model.norm (affine | per_feature | none;
             batchnorm refused with reasons)
             ([[activation-families-norm-knob]])
    75c429e  overnight: the README didactic pass (vocab box, Run-it
             split, workstation-assumption sweep)
             ([[readme-run-it-definitions]]) + weight decay by module
             role ([[weight-decay-only-weight-matrices]])
    378dc95  NPCE from the YAML (top-level pce: block)
             ([[npce-yaml-wiring]])

## IN FLIGHT (the uncommitted tree)

Save schema v2 ([[save-schema-resolved-config]]): CORE + riders 1-5
implemented; Architect re-audit = ACCEPTED WITH TWO DELTAS, the
Implementer owes the micro-fix:
- D-SV1: compile_mode's recipe fallback duplicates make_model's
  default -> ONE shared DEFAULT_COMPILE_MODE constant consumed by
  both.
- D-SV2: resolved_train["eval_bs"] reads a key nothing writes ->
  the loop stashes data["eval_bs"] = eval_bs; GSV-A gains the VALUE
  case.
Then ONE combined commit (riders + core; sentence in the note's
verdict). Also staged untracked: compute_data_vectors/
dataset_generator_lensing.py (the user's generator, verbatim,
py_compile-clean — its unit is next; EXCLUDE from the save commit:
`git add -A -- ':!compute_data_vectors'`).

## The queue after the save commit

1. The generator import + the "Generating the training set" README
   appendix ([[compute-data-vectors-import]]) — script already
   staged; the appendix teaches the tempered posterior
   (T / maxcorr / boundary), the TWO sampling modes as peers
   (tempered + uniform w/ the still-needs-temp subtlety), the
   output contract; carries the two prose docs the save unit left
   (train_single --save paragraph; README outputs sentence).
2. The cobaya Theory adapter ([[cobaya-theory-adapter]]) —
   EmulatorPredictor in emulator/inference.py + the thin
   cobaya_theory/emul_cosmic_shear.py; ord/extrapar/duplicated
   architectures DIE; fast_params passthrough; the geometry is the
   requirements authority; decision points: dv shape vs the C
   likelihood, in-theory M calibration, rebuild's pce return.
Either order (both handoff-ready). Then the program's arc is
complete: generate -> train -> save -> sample.

## Standing rules born this session (each also in its home note)

- ARTIFACTS NEVER TRUST DEFAULTS (user verbatim; the consumed-view
  doctrine's three legs: displays RENDER, artifacts PERSIST, loaders
  TRUST ONLY — [[save-schema-resolved-config]]).
- The unified GitHub-math rule: NO backslash + ASCII punctuation in
  math (GitHub eats it); no \begin environments; no line-start
  Markdown tokens in $$ blocks; no whitespace-adjacent $ spans;
  fences exempt. The five-rule scanner runs in every doc gate.
- Every ### subsection documenting YAML knobs carries its own code
  block; AI-Usage is ALWAYS the main README's last section;
  every-file's-functions is ALWAYS the code map's last section.
- The note is the spec of record (the relayed handoff block may lag
  it — the Implementer executes the NOTE; precedent D-DOC9, the
  three-addenda micro-unit, riders 3-5).
- A mid-unit STOP at a valid tree with an executable blueprint is
  the CORRECT failure mode for a budget wall (commended, never
  penalized).

## The workstation board (one session closes it; order matters)

GM-C FIRST (its golden pre-EMA leg needs `git checkout 46ec5e1`),
then the standing gates (GM-D, the production --diagnostic run
closing G-F/GN-F/GS-D/GT-C/G1, GP-D, GH-E, GE-C, GB-C, GL-D,
GBA-C, GME-C, item-27, GT-B optional) PLUS this week's legs:
GFT-C (joint phase 2), GHA-F (pinned-head gated_power), GAN-C
(tanh + per_feature / affine), GWD-C (the decay census), GPC-C
(NPCE residual + ratio + refit smoke), GSV-C (the bitwise + drift
proof — THE acceptance of the save unit), and later GCT-C (the
MCMC parity probe). Sync ritual first: the workstation checkout
must show the latest commit.

## Science thread (unchanged, waiting on the board)

berhu_capped head vs sqrt baseline (attribution); bs + EMA; the
activation bake-off now extensible (--activations ... relu tanh +
the norm knob = the honest classic baseline); final window values;
NEW: the NPCE runs the user is undeterred about ("past failures on
low T do not discourage me").

## Lessons owned this session (the Architect's own harness bugs)

Exclusion grep filters ate true positives twice ("in parallel"
swallowed "live in parallel/"; the fifteen-vs-fourteen miscount);
the em-dash slugger false-flagged working anchors; a regex matched
the FIRST xi display (nla) instead of the TATT one; a harness
tested absence-handling at the wrong contract layer (the guard
lives at the call site). Pattern: verify the harness against a
known-good case before trusting its verdict on the unknown one.
