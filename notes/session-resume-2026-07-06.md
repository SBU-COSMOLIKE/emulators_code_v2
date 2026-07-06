---
name: session-resume-2026-07-06
description: "Post-compact pickup pointer for the 2026-07-05/06 Opus session on branch amazing-keller: the package style/doc audit plus three data-cut features. Consolidates the commit state (two committed, two uncommitted-in-worktree), the full WORKSTATION-DEFERRED validation queue (item 27 + G1 runtime import + G-F + GN-F + GT-B/GT-C), the Architect re-audit still owed on the two uncommitted features, and the Mac exec-extract test method that gated everything. Details live in the four per-feature notes."
metadata:
  node_type: memory
  type: project
---

# Session resume (2026-07-05/06, Opus on amazing-keller)

One Implementer session (dual-agent, Fable = Architect) did a whole-package
style/doc audit and three physical-cut features, each with its own note and
gate evidence. This is the consolidated pickup after a compaction.

## What was done (each has its own note + resume + raw gate evidence)

1. Package style/doc audit -> [[audit-package-style-2026-07-05]] (items 1-29,
   D1-D6). COMMITTED (`e2394ed`).
2. omegamh2 + omegamh2*ns window cuts -> [[omegamh2-ns-product-cuts]]
   (phys_cut_idx quantity table). COMMITTED (`a0cd132`).
3. data.param_cuts nested block + cut -> omegabh2_hi rename ->
   [[param-cuts-nested-block]]. UNCOMMITTED (in the worktree).
4. Triangle cut-shading, all four windows -> [[triangle-cut-shading-all-windows]]
   (plotting.py). UNCOMMITTED (in the worktree).

## Commit state (git log on amazing-keller)

- `a0cd132` window cuts, `e2394ed` audit, then the base `13d832e`.
- Working tree = features 3 + 4 together (they share plotting.py /
  experiment.py / data_staging.py / train_single / README / the 3 YAMLs;
  10 files). The user commits each feature as its own unit; the Implementer
  never commits.

## Owed to the Architect (re-audit)

- Re-audit features 3 and 4 on THIS branch (amazing-keller), not the
  Architect's stale sleepy-lumiere checkout. Features 1 and 2 were already
  Architect-verified before their commits.

## WORKSTATION-DEFERRED validation queue (needs torch + cosmolike +
## matplotlib/getdist, none on the Mac dev box)

Run these in one workstation session (see [[test-workstation-gpus]] for how
to pin the right GPU):

- item 27 (audit): geometries_output.py calls `ci.init_probes` twice
  (~lines 209 and 217); resolve with chi2 A/B evidence (identical with /
  without the second call -> delete the duplicate; else a comment citing what
  broke). Never resolved statically.
- G1 runtime leg (audit): `python -c "import emulator, emulator.IA,
  emulator.PCE, emulator.parallel"` clean on a torch machine (py_compile
  already clean tree-wide; the import needs torch + cosmolike).
- G-F (window cuts): one short training with a tight omegamh2 window; the
  pool shrinkage matches the load banner's per-window kept/total.
- GN-F (param_cuts nesting): one load with the nested `param_cuts:` block
  shows the normal banner (content byte-identical to the old flat layout).
- GT-B (triangle shading): a synthetic-sample triangle with all four windows
  active has contourf artists on exactly the coverage-table panels (assert on
  the axes' artist lists), same rgba, plus the omh2-marginal axvspans.
- GT-C (triangle shading): regenerate the diagnostics PDF for the flagged run
  (diagnostic_rescnn_t16_ntrain25000-style); grey now adjoins the omh2
  marginal cliff at 0.20 and the (ns, omh2) 0.17 corner.

## The Mac test method (why the Mac gates could run at all)

The Mac dev python has numpy + stdlib but NOT torch / cosmolike / matplotlib /
getdist / pyyaml, and the package modules import those at load. So the gates
ran by EXTRACTING the pure functions from source (ast-parse, take the
function/const nodes by name, `exec` their source span into a numpy-only
namespace) and testing those in isolation, plus AST/tokenize scans and
`py_compile` (which compile without importing). This is the durable enabler
for Implementer-side validation on the Mac; captured in
[[dev-machine-mac-m2-32gb]]. The scan/rewrite/astdiff/codeskel tools and the
per-feature test harnesses lived in the session scratchpad (ephemeral); the
method, not the files, is what carries over.
