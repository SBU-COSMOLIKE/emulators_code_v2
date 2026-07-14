# notes/ index

Consolidated 2026-07-11: ~85 topic notes rewritten into the compact topic and
audit set below so any model (or human) can orient fast. Every retired note
survives in git history (`git log --follow notes/<old-name>.md`); the
delta IDs preserved in these files are the search keys.

Read in this order for a cold start: (1) this index,
(2) the durable correctness registry in
red-team-audit-and-didactics-2026-07-13.md, (3) project-and-history.md, then
the topic file your task touches. Use state-2026-07-11-and-next.md for
chronology and routing, not as a live queue: its opening snapshot is stale and
its later blocks preserve decisions rather than one canonical current order.
Current sequencing comes from `gates-and-board.md`, "The consolidated
DIDACTICS execution handoff" (D1--D10), and must be checked against the
relevant topic note before implementation.

Hard communication rule: the substantive exchange between Fable, the
Implementer and the Red Team is written under `notes/` before a chat relay is
sent. Chat handoffs are summaries that cite the durable note. See
`conventions-and-workflow.md`, "Notes-first inter-agent communication." The
same rule governs `notes/mailbox/` dispatches: a mailbox file is a routing
summary, and a mailbox-started turn writes its outbound block to the next
numbered mailbox file after recording the substance under `notes/`.

- [State + chronological ledger](state-2026-07-11-and-next.md) — the
  historical run trail, adjudication batches, queue changes, retractions, and
  topic-note routing. Do not promote any one historical run snapshot or queue
  paragraph to current state.
- [Project + history](project-and-history.md) — the goal, the
  development arc by phase, the family-pattern recipe (what a new
  output family adds), the program-level lessons.
- [Training stack](training-stack.md) — losses (sqrt/chi2/berhu
  ladder + roughness), the shared anneal family, phase blocks +
  demotion, EMA + the snapshot invariant, consumed-view banners,
  sizing (absolute counts, derived eval bs, weight-decay allowlist),
  the loud no-alias migration pattern; open red-team details include
  the dead validation-safe chunk, the unbounded wide-output
  diagnostics path, and activation-bakeoff process liveness.
- [Models + designs](models-and-designs.md) — ResMLP/ResCNN/ResTRF,
  the correction-head philosophy, zero-init identity discipline,
  factored IA, activations/norms, FiLM, NPCE, the science doctrine
  (sample efficiency, coverage floor), the CLOSED-experiment ledger.
- [Artifacts + inference + warm starts](artifacts-inference-warmstart.md)
  — schema v2 (never-trust-defaults), rebuild, EmulatorPredictor, the
  five cobaya adapters (python_path trap incl.), fine-tuning (FTW),
  transfer (TPE, refine, anchors), the geometry folder (GEO,
  D-GEO5 shims retired), plus the open real-Cobaya dependency-routing
  defects hidden by stubbed adapter gates and the open geometry-state /
  covariance-validation contract.
- [Families: scalar + CMB](families-scalar-cmb.md) — SPE (closed,
  the lesson bank) and CME (covinv ruling, amplitude law, covariance
  script, roughness, diagnostics dispatch) + the D-CM12 SPEC AWAITING
  AUDIT and D-CM13 IMPLEMENTED (heads on every coordinate family,
  user-ordered 2026-07-11).
- [Families: background + matter power](families-background-mps.md)
  — BSN (two-regime, imposed distances, flat-only, the Simpson
  finding) and MPS (correction-to-syren, D-MP2-A base-on-disk, the
  vendored syren/, EMUL2, MPS-DIAG).
- [Data generation + cuts](data-generation-and-cuts.md) — the four
  generators on generator_core + the covariance script, tempered/
  uniform sampling, the output contract, staging/memmap, the
  param_cuts windows and the coverage-cut lesson; open checkpoint-set
  integrity covers manifest membership, append publication, and the
  destructive load-error fallback.
- [Gates + the board](gates-and-board.md) — the harness, identity/smoke
  philosophy, dead-network rule, resume/evidence design, manifest population,
  and run history. `gates/board.py` plus `python3 gates/run_board.py --list`
  are authoritative for the current gate set; prose counts are snapshots.
  1b hardening COMPLETE (9/9 + item-7: 18 coverage, 16 runtime-loaders +
  data-read leaves, 19 owner resolvers + run-time refusal, 26 lineage) plus the
  27/28 machinery follow-up (tracked-driver watch, shared stale-member surface);
  QUEUE 2 OPENS. Remaining texnotes/D1 prose deferrals only.
- [Conventions + workflow + environment](conventions-and-workflow.md)
  — the Python/docs/README/plots/terminal/YAML house rules, the
  dual-agent workflow, git discipline, the Mac evidence pattern,
  machines and ROOTDIR.
- [User didactics + Python voice](user-didactics-and-python-voice.md)
  — who the reader is (C coder; cosmologist audience), how she likes
  to be taught (show-never-describe, define-or-drop, run-first), and
  the code register with her own quotes and before/after shapes; the
  Implementer reads it before writing code or docs.
- [Whole-package style audit (2026-07-05)](audit-package-style-2026-07-05.md)
  — dated pre-consolidation Architect record: whole-tree audit of
  emulator/ + the five drivers + README + example_yamls (structural
  axes PASS; the fix list was handed off 2026-07-05 and largely
  executed by the later doc/POL sweeps — read it as history + method,
  not an open queue). Side-finding: the june2026/claude_skills copy of
  pytorch-teaching-style was stale; the live skill lives in
  ~/data/claude_skills/.
- [Red-team Implementer handoff (2026-07-13)](red-team-implementer-handoff-2026-07-13.md) — single-page 45M inventory + per-unit commits/files/gates + the "audit hardest here" guidance (86-90 vs the Architect specs now on main, and the five 45M-72 design decisions to rule on); the review brief for the Architect audit pass. Landed on origin/main via merge 8ce72a9.
- [Red-team audit + durable handoff registries (2026-07-13)](red-team-audit-and-didactics-2026-07-13.md) — independent review of the Implementer landing; binding evidence-map rulings; the canonical pre-45M unit crosswalk; complete 45M/20M/RT/BLOAT routing with explicit unused-id tombstones; the DIDACTICS-42--100 registers; and the continuing 25M correctness index. Future Red Team handoffs are appended here or to their existing topic note before their chat copy is sent.
- [Mailbox daemon incident (2026-07-14)](mailbox-daemon-incident-2026-07-14.md) — the 0014 placeholder body reached a live Opus turn because the running `--watch` process (lock pid 45327, started 00:52) predates its own fix (`55eb256`, 00:53:06): a daemon hardening commit is inert for the loop already running. Plus two more transport defects (refusals are never quarantined; `next_seq()` collides — `done/` already holds a duplicate 0008 pair) with ready-to-apply patches. RESTART THE WATCHER FIRST — until then no committed guard is protecting the loop.
