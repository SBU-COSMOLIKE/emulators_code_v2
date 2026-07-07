---
name: session-status-2026-07-06b
description: "Architect session state at the SECOND 2026-07-06 compaction. Mac side FULLY CLOSED: nine feature commits landed this session (chain 7ebc061 EMA -> 76ef641 phase-blocks -> 64786a7 berhu -> dcbaf9f loss-nesting -> a043e28/c328da9 berhu-anneal+D-L1 -> 5b65fd5 D-L1v3+D-P2 -> 07e4564 D-P2v2 -> 8bb5484 ema-anneal+per-phase-ema), every one Architect-audited; working tree clean. Remaining: user merges+SYNCS THE WORKSTATION (the step that broke two runs today), fixes the production YAML (lr block with bs_base moves from trunk: to top level; nepochs must exceed trunk_epochs before running rescnn), then ONE workstation session closes the whole gate board (GM-C FIRST, needs checkout 46ec5e1 for its pre-EMA golden leg), then the science (berhu-vs-sqrt head, bs+EMA experiments, final window values). Read this first to resume; per-feature detail in the linked notes."
metadata:
  node_type: memory
  type: project
---

# Session status at compaction 2 (2026-07-06, Architect)

Everything below is on branch claude/amazing-keller-e798b6. The tree is
CLEAN at 8bb5484 — all nine of this session's feature units are
committed and Architect-audited. No open handoffs, no open deltas.

## The day's commit chain (each note carries spec + evidence + audit)

    7ebc061  weight EMA coupled to snapshot/rewind
             ([[weight-ema-snapshot-coupled]]; + eval-bs D-E1/D-E2, D-M1)
    76ef641  phase blocks: nested lr + per-phase scheduler
             ([[phase-blocks-nested-lr-scheduler]])
    64786a7  loss_mode berhu + berhu_capped ([[loss-mode-berhu]])
    dcbaf9f  train_args.loss nesting, D-B1 machinery deleted
             ([[loss-block-nesting]])
    a043e28 + c328da9  berhu anneal (sqrt -> berhu blend)
             ([[berhu-anneal-schedule]]; c328da9 carried D-L1 v2)
    5b65fd5  D-L1v3 (mode-named berhu block ACCEPTED, not errored)
             + D-P2 capability-aware banner
    07e4564  D-P2v2: consumed-view banner, class-owned describe_spec,
             head_block attr, ARCH_HEAD retired
             ([[banner-prints-consumed-view]] = the STANDING DIRECTIVE)
    8bb5484  ema.anneal (horizon schedule) + per-phase ema with null
             opt-out ([[ema-anneal-schedule]])

Earlier the same day (before the first compaction note
[[session-status-2026-07-06]]): 9950692 D-1 guard, 906528c n_train/
n_val, 46ec5e1 eval-bs derivation, 4471539 sweep guards + D-P1,
2b7a2af resolve_phase_args, plus the audit/cuts/nesting/triangle
work of 2026-07-05/06a.

Also committed somewhere in the chain: activation_functions_teaching.nb
(repo root) — the students' Mathematica notebook on the H /
multigate / power / gated_power activation family, kernel-validated.

## Immediate user actions (before the workstation session)

1. Merge to the main checkout if not done (`cd ../../..;
   git merge claude/amazing-keller-e798b6`), then SYNC THE WORKSTATION
   — its checkout must show 8bb5484 at `git log -1`. Stale workstation
   builds broke two runs today (fingerprints: an error message's
   whitelist contents identify the commit).
2. Production YAML fixes (from the last exchanges):
   - the `lr:` block (lr_base, bs_base, warmup_epochs) moves from
     `trunk:` to the TOP level — bs_base is the run-global sqrt-rule
     anchor and is rejected inside a phase lr (by design, both paths);
     phase blocks are DIFFS against the top level, not containers;
   - optionally move scheduler/trim/focus to top level too (trunk
     inherits; head keeps its overrides);
   - `ema:` stays inside `trunk:` (trunk-only averaging, now legal);
   - nepochs (1000) must exceed trunk_epochs (1000) before this YAML
     ever runs on a two-phase model (fine on resmlp: both demoted).

## The workstation session (ONE session closes the whole board)

Order constraint: GM-C FIRST — its golden "pre" leg needs
`git checkout 46ec5e1` (pre-EMA) before later features muddy the
comparison. Then in any order (recipes embedded in each note):

    GM-C   EMA off-mode golden (bit-identical epoch lines)
    GM-D   EMA on-mode smoke (banner; rewind: EMA jumps WITH raw)
    -- one production train_single --diagnostic run closes:
    G-F    window-cut smoke        GN-F  param_cuts banner
    GS-D   sizes banner            GT-C  triangle shading PDF
    G1     runtime import leg
    GP-D   resmlp demotion run + rescnn+nla control
    GH-E   head scheduler patience-10 cadence + no-blocks golden
    GE-C   eval partition-invariance (torch-only script)
    GB-C   berhu: unbound _reduce script + mixed-config run vs the
           sqrt-head baseline (THE physics readout)
    GL-D   loss-schema golden equivalence
    GBA-C  berhu-anneal golden + hold-boundary smoke
    GME-C  ema-anneal golden + live-point smoke
    item-27 duplicate ci.init_probes A/B (geometries_output; the
           oldest open item)
    GT-B   triangle artist smoke (optional)

The Architect closes every gate in one pass on the pasted outputs +
the regenerated triangle PDF.

## Then the science (what the infrastructure was built for)

- berhu_capped head vs sqrt baseline (soften the head focus for clean
  attribution — berhu carries the tail role).
- Batch-size + EMA: does bs 32-64 + EMA close the tail further, now
  that the derived eval batch stopped punishing small bs (the
  launch-bound analysis: wall time ~ steps, per-step 2.05 ms flat;
  the mid-run x1.5 drift was a CONCURRENT job's checkpoint saves, not
  clocks — GPU steady at 1807 MHz).
- Final production window values from the cut forensics (current
  trials: omegabh2 (0.005, 0.035), omegam2h2 (0.02, 0.075), omegamh2
  (0.05, 0.20), omegamh2ns (0.10, 0.17)).

## Session rules and lessons added today (each in its own note)

- [[banner-prints-consumed-view]] — STANDING DIRECTIVE: displays render
  the resolved/consumed config via the SAME functions execution uses;
  every new config feature's spec states its banner rendering; every
  audit checks display surfaces (GP-B missed print_design — owned).
- The D-B1 arc: flat schema -> inheritance patch -> DELETED by the
  user's localization directive ([[loss-block-nesting]]) — structural
  fixes beat behavioral patches.
- D-L1 arc: gracious error -> superseded by ACCEPTANCE (the mode-named
  berhu block simply works) — don't build traps and apologize; make
  the natural spelling legal.
- Phase blocks are diffs, not containers (the bs_base rejection is the
  guard rail); validation identical on both paths, always.
- The eager-float exception: the EMA lerp is uncompiled, so beta(e)
  is a per-epoch Python float BY DESIGN — the one deliberate inversion
  of the trim_t/focus_t tensor discipline (comment in code).
- berhu naming ruling (recorded in [[loss-mode-berhu]]): our loss IS
  textbook BerHu in the whitened residual norm (delta = sqrt(knot)),
  applied per sample (Mahalanobis aggregate), knot in chi2 units —
  name kept, the units/aggregate caveats are the documentation duty.

## Unscheduled ideas on the shelf (recorded, not specced)

Plain-chi2-twin floor for factored-IA diagnostics; steps-denominated
trim/focus schedules (cross-bs confound); a tunable berhu cap key;
re-sync the stale pytorch-teaching-style skill copy under
june2026/claude_skills/.
