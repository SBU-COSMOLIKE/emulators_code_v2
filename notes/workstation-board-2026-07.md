---
name: workstation-board-2026-07
description: "BOARD 2026-07-07 (Architect): the single ordered list of every deferred workstation/cocoa-env verification gate, assembled so one synced workstation session can close them all. Order matters: GM-C FIRST (its golden pre-EMA leg needs git checkout 46ec5e1, then return to tip), then the standing gates (one production --diagnostic run closes G-F/GN-F/GS-D/GT-C/G1 together), then this week's legs (GFT-C freeze_trunk, GHA-F pinned head, GAN-C relu/tanh+norm, GWD-C decay census, GPC-C NPCE smoke), then GSV-C (bitwise + drift proof — THE acceptance of save-schema-v2) feeding GCT-C (parity probe + the factored save->rebuild->predict round-trip + the evaluate run vs the lsst_y1 likelihood + MCMC smoke). Sync ritual first: the workstation checkout must show 0f056e1. Each gate's full definition lives in its home note (linked); results are appended THERE, this board only tracks open/closed."
metadata:
  node_type: memory
  type: project
---

# The workstation board (2026-07): one session closes it

Every Mac-side unit through the cobaya adapter is committed
(tip 0f056e1, merged to main). What remains on the workstation
(cocoa env, NVIDIA GPUs — [[test-workstation-gpus]]) is
verification, not development. This note is the single entry
point; each gate's full definition and acceptance criteria live in
its home note, and RAW RESULTS ARE APPENDED TO THE HOME NOTE, not
here. Here only the checklist state flips.

## Sync ritual (before anything)

    git pull   # the workstation checkout must show 0f056e1 at git log -1
               # ("Cobaya Theory adapter: ...") — else STOP and sync.

## The board, in execution order

Standing gates (accumulated before this week):

1. [ ] GM-C — FIRST: its golden pre-EMA leg needs
       `git checkout 46ec5e1`; run that leg, then
       `git checkout main` before every other gate.
       Home: [[weight-ema-snapshot-coupled]].
2. [ ] GM-D — home: [[weight-ema-snapshot-coupled]].
3. [ ] The production `--diagnostic` run — ONE run closes
       G-F / GN-F / GS-D / GT-C / G1 together.
       Home: [[driver-audit-phase-sweep-guards]] (G1 also
       [[audit-package-style-2026-07-05]]).
4. [ ] GP-D — home: [[driver-audit-phase-sweep-guards]],
       [[resolve-phase-args-single-phase]].
5. [ ] GH-E — home: [[phase-blocks-nested-lr-scheduler]].
6. [ ] GE-C — home: [[eval-bs-decoupling]].
7. [ ] GB-C — home: [[loss-block-nesting]], [[loss-mode-berhu]].
8. [ ] GL-D — home: [[loss-mode-berhu]].
9. [ ] GBA-C — home: [[berhu-anneal-schedule]].
10. [ ] GME-C — home: [[ema-anneal-schedule]].
11. [ ] item-27 — home: [[omegamh2-ns-product-cuts]],
        [[param-cuts-nested-block]].
12. [ ] GT-B (optional) — home:
        [[driver-audit-phase-sweep-guards]].

This week's legs (each closes a 2026-07-06/07 unit):

13. [ ] GFT-C — freeze_trunk: false joint phase 2 actually trains
        both stacks. Home: [[freeze-trunk-joint-phase2]].
14. [ ] GHA-F — pinned-head activation (gated_power) run.
        Home: [[head-activation-per-component]].
15. [ ] GAN-C — relu / tanh + norm: per_feature / affine legs.
        Home: [[activation-families-norm-knob]].
16. [ ] GWD-C — the weight-decay census on a live optimizer.
        Home: [[weight-decay-only-weight-matrices]].
17. [ ] GPC-C — NPCE residual + ratio + refit smoke.
        Home: [[npce-yaml-wiring]].

The save->sample acceptance chain (LAST, in this order — GSV-C's
saved artifact feeds GCT-C):

18. [ ] GSV-C — THE acceptance of save-schema-v2: save ->
        rebuild_emulator -> bitwise-equal prediction on a probe
        batch, THEN the drift proof (monkeypatch code defaults,
        equality still holds). Include one FACTORED (ia:) save and
        one NPCE save so the geometry-class marker and the pce
        group both round-trip. Home: [[save-schema-resolved-config]].
19. [ ] GCT-C — the cobaya adapter acceptance: the parity probe
        (EmulatorPredictor vs the training-side eval on the same
        probe points, rtol 1e-6); the factored
        save->rebuild->predict round-trip (D-CT1's real leg); the
        example evaluate run end-to-end against the lsst_y1
        likelihood (use_emulator: 1) with the printed datavector
        compared to the training-side prediction; an MCMC evaluate
        + short-chain smoke. Rider: the MPS-float64 check is a
        DEV-MAC leg (runs there, not on the workstation).
        Home: [[cobaya-theory-adapter]].

## After the board: the science thread

berhu_capped head vs sqrt baseline attribution; bs + EMA; the
activation bake-off extended with relu / tanh + the norm knob (the
honest classic baseline); the NPCE runs (user: "past failures on
low T do not discourage me"); final window values. These are
science operations, not gates — they start once the board is green.

## Handoff (SUPERSEDED 2026-07-07: no Claude session exists on the
## workstation — the board is executed by the USER via the gates
## harness, [[gates-harness-user-run]]. The block below is kept for
## the record only; do not relay it.)

> Superseded by the committed harness `gates/run_board.py` (framework +
> all 19 gates below encoded from these home notes; the user runs
> `python gates/run_board.py`). See [[gates-harness-user-run]].

### ARCHITECT_HANDOFF
Task: close the workstation board (spec:
notes/workstation-board-2026-07.md — the ORDER is binding, GM-C's
checkout-46ec5e1 leg first and back to main before anything else;
each gate's definition + acceptance live in its home note, read
the home note before running its gate). Base: `git log -1` must
show 0f056e1 ("Cobaya Theory adapter: ...") — else STOP and sync.
Environment: cocoa env active (cosmolike + cobaya importable),
GPUs visible. Report: raw gate outputs appended to EACH home note,
the checklist here flipped, an IMPLEMENTER_HANDOFF summarizing
pass/fail per gate with any deviation declared. Do not commit:
print the suggested commit command (explicit paths).
### END

## Status

BOARD ASSEMBLED 2026-07-07 (Architect), all gates OPEN. The
MPS-float64 rider (GCT-C) runs on the dev Mac separately.
