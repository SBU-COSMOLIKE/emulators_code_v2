# notes/ index

Consolidated 2026-07-11: ~85 topic notes rewritten into the ten files
below so any model (or human) can orient fast. Every retired note
survives in git history (`git log --follow notes/<old-name>.md`); the
delta IDs preserved in these files are the search keys.

Read in this order for a cold start: (1) this index,
(2) state-2026-07-11-and-next.md, (3) project-and-history.md, then
the topic file your task touches.

- [State + what must still be tested](state-2026-07-11-and-next.md) —
  where the code stands, what landed 2026-07-11, the ordered
  user-run test queue (board, artifacts, EMUL2, spec audits).
- [Project + history](project-and-history.md) — the goal, the
  development arc by phase, the family-pattern recipe (what a new
  output family adds), the program-level lessons.
- [Training stack](training-stack.md) — losses (sqrt/chi2/berhu
  ladder + roughness), the shared anneal family, phase blocks +
  demotion, EMA + the snapshot invariant, consumed-view banners,
  sizing (absolute counts, derived eval bs, weight-decay allowlist),
  the loud no-alias migration pattern.
- [Models + designs](models-and-designs.md) — ResMLP/ResCNN/ResTRF,
  the correction-head philosophy, zero-init identity discipline,
  factored IA, activations/norms, FiLM, NPCE, the science doctrine
  (sample efficiency, coverage floor), the CLOSED-experiment ledger.
- [Artifacts + inference + warm starts](artifacts-inference-warmstart.md)
  — schema v2 (never-trust-defaults), rebuild, EmulatorPredictor, the
  five cobaya adapters (python_path trap incl.), fine-tuning (FTW),
  transfer (TPE, refine, anchors), the geometry folder (GEO,
  D-GEO5 shims retired).
- [Families: scalar + CMB](families-scalar-cmb.md) — SPE (closed,
  the lesson bank) and CME (covinv ruling, amplitude law, covariance
  script, roughness, diagnostics dispatch) + the D-CM12/D-CM13
  SPECS AWAITING AUDIT.
- [Families: background + matter power](families-background-mps.md)
  — BSN (two-regime, imposed distances, flat-only, the Simpson
  finding) and MPS (correction-to-syren, D-MP2-A base-on-disk, the
  vendored syren/, EMUL2, MPS-DIAG).
- [Data generation + cuts](data-generation-and-cuts.md) — the four
  generators on generator_core + the covariance script, tempered/
  uniform sampling, the output contract, staging/memmap, the
  param_cuts windows and the coverage-cut lesson.
- [Gates + the board](gates-and-board.md) — the harness the user
  runs, the 32 gates, identity/smoke philosophy, dead-network rule,
  the run-history table and its lessons.
- [Conventions + workflow + environment](conventions-and-workflow.md)
  — the Python/docs/README/plots/terminal/YAML house rules, the
  dual-agent workflow, git discipline, the Mac evidence pattern,
  machines and ROOTDIR.
