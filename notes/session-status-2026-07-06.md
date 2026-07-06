---
name: session-status-2026-07-06
description: "Architect session state at compaction (2026-07-06): what is committed, what awaits the user's commit (two audit-accepted features stacked uncommitted on amazing-keller + the split-commit block), the consolidated workstation queue (one train_single run closes G-F + GN-F + G1-import + GT-C; item-27 A/B and GT-B separate), the production window values under trial, and the t16_ntrain25000 diagnostics readout (coverage-limited). Read this first to resume; per-feature detail lives in the linked notes."
metadata:
  node_type: memory
  type: project
---

# Session status at compaction (2026-07-06, Architect)

Everything below happened on branch claude/amazing-keller-e798b6 (worktree
.claude/worktrees/amazing-keller-e798b6). The sleepy-lumiere worktree is
stale and disposable.

## Committed (by the user; e2394ed, a0cd132)

1. Whole-package style/doc audit fixes, D1-D5 closed
   ([[audit-package-style-2026-07-05]]).
2. The omegamh2 + omegamh2ns window cuts, gates A-E
   ([[omegamh2-ns-product-cuts]]).

## Awaiting the user's commit (both Architect-accepted, stacked uncommitted)

1. data.param_cuts nesting + omegabh2_cut -> omegabh2_hi rename
   ([[param-cuts-nested-block]], GN-A..E verified).
2. Triangle cut shading, all four windows, same-grey superposition
   ([[triangle-cut-shading-all-windows]], GT-A verified).

Split-commit block (triangle = plotting.py + its note; nesting = the rest):

    cd .claude/worktrees/amazing-keller-e798b6
    git add -A
    git reset emulator/plotting.py notes/triangle-cut-shading-all-windows.md
    git commit -m "Nest cut keys under data.param_cuts, rename omegabh2_cut -> omegabh2_hi (loud migration, gates GN-A-E Architect-verified)"
    git add emulator/plotting.py notes/triangle-cut-shading-all-windows.md notes/MEMORY.md
    git commit -m "Shade all four cut windows on the diagnostics triangle (per-window same-grey superposition, GT-A Architect-verified)"
    cd ../../..
    git merge claude/amazing-keller-e798b6

## Workstation queue (torch + cosmolike; mostly ONE run)

1. One short train_single run with the NESTED param_cuts YAML, a tight
   omegamh2 window, small nepochs, --diagnostic. Closes at once:
   G-F (window smoke), GN-F (nested-block load banner), the runtime
   `import emulator...` leg of G1, and GT-C (regenerated triangle: grey
   must adjoin the omh2-marginal cliff and the sheared (ns, omh2) corner).
2. Item 27 ([[audit-package-style-2026-07-05]] P6): rebuild the output
   geometry with the second ci.init_probes(possible_probes=probe) in
   geometries_output.py commented out; compare state() tensors
   (dest_idx / Cinv / center) + one fixed-seed random-dv chi2.
   Identical -> delete the duplicate; different -> keep + why-comment
   citing the evidence. One-line follow-up commit either way.
3. GT-B (artist-level render smoke) — optional if GT-C looks right.

The Architect closes all remaining gates in one pass on the pasted outputs
+ the regenerated PDF.

## Production context (user's current trial)

- YAML (post-nesting schema): param_cuts with omegabh2 (0.005, 0.035),
  omegam2h2 (0.015, 0.08), omegamh2 (0.05, 0.20), omegamh2ns (0.10, 0.17).
  The omegamh2_lo = 0.05 side is inactive in practice (prior floor ~0.09).
- t16_ntrain25000 rescnn+nla diagnostics readout: coverage-limited —
  spearman(knn_dist, log dchi2) +0.261; frac>0.2 dense 0.126 vs sparse
  0.328; hardness joint log-linear R^2 0.361 (the direction behind the new
  windows, [[omegamh2-ns-product-cuts]]); floor panel correctly skipped
  (param-aware factored-IA loss has no per-sample target in model space —
  a plain-chi2-twin floor for factored runs is a possible future addition,
  not specced).
- Two-phase timing signature confirmed healthy on the 3060: trunk phase
  ~0.1 s/epoch (head bypassed + CUDA graphs), head phase ~0.4 s/epoch
  (trunk forward + W_fd/W_df matmuls + bandwidth-bound conv); the old
  ~0.7 s/epoch predates the 04t launch-bound fixes
  ([[nla-as-design-spec]] 04t) and was a single-phase joint run.

## Session rules added (also in auto-memory + [[dual-fable-opus-workflow]])

- User-only commits; command blocks always carry a concrete
  `git commit -m "..."` sentence AND the merge-to-main steps.
- Any YAML-key change is reported with a paste-ready block-context snippet.
- Handoffs paste raw scan outputs, not summaries (three over-claims caught:
  caps "0" vs 1, dashes "0" vs 3, PS "covered" vs five drivers empty);
  interface changes must be declared as deviations even when good
  ((kept_idx, report) tuple; consumer renames were declared properly after
  the first slip).
- Width-vs-paren-align precedence ruling recorded in
  [[py-module-style-conventions]] (90 cols hard; one-item-per-line hanging
  fallback; one style per file).
- The live pytorch-teaching-style skill is
  ~/data/claude_skills/pytorch-teaching-style/SKILL.md; the copy under
  june2026/claude_skills/ is stale (Jun 18) — user should re-sync.
