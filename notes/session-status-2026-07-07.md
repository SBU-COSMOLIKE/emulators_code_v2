---
name: session-status-2026-07-07
description: "Session state at the 2026-07-07 compaction (READ FIRST to resume). Since the last snapshot ([[session-status-2026-07-06b]], tip 8bb5484) five more units landed and are committed on claude/amazing-keller-e798b6: 2d2f68d doc-audit D-DOC1..9, then the THREE-UNIT head-activation cycle ebd9869 freeze_trunk (unit A) -> a95f8df per-head activation (unit B) -> ddb6c98 precedence appendix (unit C), then fc4655a the README YAML chapter. Every unit Architect-audited (A accepted; B 30/30; C GPR-A; the YAML chapter GYC-A..E). Tree clean except two uncommitted NOTES (this index + the Architect's next spec readme-reorg-two-readmes). Mac side fully closed. Remaining: the workstation gate board (now + GFT-C unit A and GHA-F unit B), the production-YAML fixes, then the science. Next queued work: [[readme-reorg-two-readmes]] (split the one README into two). Open ruling: the optimizer+lr+scheduler one-section grouping in README section 11 (user-veto-able)."
metadata:
  node_type: memory
  type: project
---

# Session status at the 2026-07-07 compaction (read first to resume)

Branch `claude/amazing-keller-e798b6`. Tree is CLEAN at `fc4655a` — every
feature/doc unit below is committed and Architect-audited. The only
uncommitted files are notes (this index + the Architect's next spec). No
open handoffs, no open code deltas.

Roles this session: Opus = Implementer, Fable 5 = Architect; the user
relays the handoff blocks and runs every git commit/merge/push.

## The commit chain since the last snapshot (each note = spec + evidence + audit)

    2d2f68d  documentation truth pass D-DOC1..9
             ([[audit-docs-2026-07-06]]): print_design banner prints the
             resolved loss + ema blocks (the one code edit); eight-key
             phase enumerations; README EMA/berhu/validator/demotion/
             derived-eval-bs coverage; the TRF MLP token-width pin.
    ── the THREE-UNIT head-activation cycle (units share files -> SEQUENTIAL
       commits; A lands before B because B's license check reads it) ──
    ebd9869  UNIT A: train_args.freeze_trunk (joint phase 2)
             ([[freeze-trunk-joint-phase2]]) — set_train_phase("head" if
             freeze else "joint") via a role/name split; demotion + sweep
             guards mirror trunk_epochs; banner says joint; closed D-DOC2b
             obs2/obs3. Gates GFT-A/B (Mac).
    a95f8df  UNIT B: per-head-component activation
             ([[head-activation-per-component]]) — model.cnn/.trf.activation
             pins the head family (absent = share the trunk's); four seams
             (head_act ctor kwarg + build_specs {type,n_gates} value
             special-case); the head: activation: alias + trunk: asymmetry
             error; the frozen-trunk-head-phase LICENSE at build_specs; the
             flag-vs-pin + pinned-head-sweep startup warnings; closed
             D-DOC2b obs1. Gates GHA-A..E (Mac).
    ddb6c98  UNIT C: README precedence appendix
             ([[readme-precedence-appendix]]) — "who wins when settings
             collide" (sections A-G); the ram_frac row CORRECTED against the
             tree (only sweep_ntrain/sweep_hyperparam/bakeoff force 0, the
             tune workers divide by n_workers). Gate GPR-A.
    fc4655a  the README YAML chapter ([[readme-yaml-chapter]]) — sections
             7-16 document every YAML block with equations (10 verified
             verbatim vs the tree) + block-style examples; section 6 tour
             shrunk to a pointer; appendices renumbered 17-21 with
             every-file's-functions LAST. Gates GYC-A..E; Architect
             re-audit ACCEPTED no deltas (61 anchors resolve, vote column
             re-derived). The latent $$-underscore "Double subscripts" bug
             it left at sections 10/11/14 (steps_per_epoch etc.) is now
             FIXED in the reorg (GRO-G): symbols + legends, the sections
             are now main 5/6/9.

Prior chain (through the last snapshot): [[session-status-2026-07-06b]].

## Immediate user actions (unchanged from the last snapshot, still owed)

1. Merge to the main checkout + SYNC THE WORKSTATION to `fc4655a` (stale
   builds broke runs before; an error message's whitelist fingerprints
   the commit).
2. Production-YAML fixes (still open): the `lr:` block with `bs_base`
   belongs at the TOP level, not in `trunk:` (phase blocks are diffs, not
   containers); `nepochs` must exceed `trunk_epochs` before a two-phase
   run. The new precedence appendix (section 20) + the YAML chapter
   (sections 7-16) now document all of this.

## Workstation gate board (ONE session closes it)

Order constraint unchanged: GM-C FIRST (its golden "pre" leg needs
`git checkout 46ec5e1`, pre-EMA). Then, in any order, the recipe embedded
in each note:

    GM-C / GM-D    EMA off/on golden + smoke
    GFT-C          freeze_trunk: golden absent-key byte-identity + a joint
                   run (banner "joint", loss-continuous handoff, phase-2
                   epoch time above the frozen control)  [NEW, unit A]
    GHA-F          per-head activation: golden no-key byte-identity + a
                   restrf + trf.activation gated_power run (banner shows
                   the pin, head param count rises vs H, handoff continuous;
                   the flag-vs-pin warning; the license error)  [NEW, unit B]
    GB-C / GL-D / GBA-C / GME-C   berhu / loss-schema / berhu-anneal /
                   ema-anneal goldens + smokes (the physics readout is
                   GB-C: berhu head vs the sqrt baseline)
    G-F / GN-F / GS-D / GT-C / G1 / GP-D / GH-E / GE-C   the older board
                   (one production train_single --diagnostic run closes
                   several; see [[session-status-2026-07-06b]] for the full
                   list + which gate each leg feeds)
    item-27        duplicate ci.init_probes A/B (geometries_output; oldest
                   open item)

Units C + the YAML chapter are DOCUMENTATION-ONLY (no workstation leg).

## Next queued work

- [[readme-reorg-two-readmes]] — IMPLEMENTED 2026-07-07 (Opus, base
  fc4655a), UNCOMMITTED, awaiting Architect re-audit. Split the one root
  README into two: main README = Run it first + the YAML chapter
  renumbered 2-11 + Pipeline demoted to appendix 12 + chi2/activations/
  precedence 13-15 + AI-Usage last (verbatim two sentences); NEW
  emulator/README.md code map = Layout / What each file does / Change X
  -> edit Y / Variants / every-file's-functions last. `git rm -r
  emulator/parallel/` staged + all three parallel/ docstring mentions
  cleaned (the spec's "five, complete" list missed two .py ones —
  declared). The GitHub math-render fix landed at the loss / optimizer /
  ema sections (now main 5 / 6 / 9): symbols + legends, no underscore
  identifiers inside $$ — the "Double subscripts" ema bug is gone. Gates
  GRO-A..G all green (11 verbatim moves byte-identical; AST code-identical
  on both touched .py; py_compile clean). Resume state + raw evidence in
  the note.

## Open ruling

- README section 11 groups optimizer + lr + scheduler into ONE section
  (Architect-recommended; the user listed lr standalone). USER-VETO-ABLE
  — split into three on request. Flagged in the YAML-chapter handoff.

## Lessons added this cycle (each in its own note)

- Sequential-commit discipline: when units share files and must be
  separate commits, land them one at a time (commit A, then implement B
  on the committed base). Path/hunk staging cannot cleanly separate shared
  files after the fact.
- freeze_trunk role/name split: the pass ROLE stays "head" (selects
  head_opts, drives the best-epoch restore, labels the banner) while
  set_train_phase receives the model NAME ("joint" when not frozen) — one
  local, no optimizer-path change (the freeze already acts via
  requires_grad).
- Per-head activation is a CONSTRUCTION knob, not a training diff: the
  head is identity through phase 1, so the head component's activation IS
  the head phase's — model.cnn/.trf.activation, resolved in build_specs,
  licensed by a frozen-trunk head phase. The phase whitelist stays the
  eight training keys; the head: activation: alias is a ninth, head-only,
  non-training key.
- Displays reuse resolution ([[banner-prints-consumed-view]] still binds):
  describe_spec renders the head activation for free; the flag-vs-pin
  warning is built where the flag's explicitness is known (from_config).
- Standing README rule (recorded): "Appendix: every file's functions" is
  ALWAYS the last appendix; it owes a completeness update whenever new
  public defs land (this cycle added the four unit-B helpers).
- A verify-at-build spec row can be wrong: GPR-A caught the ram_frac row
  (the tune workers divide, they do not force 0). Cross-check every
  documented rule against the tree, correct + declare.

## The science (what the infrastructure was built for) — unchanged

berhu_capped head vs the sqrt baseline (soften the head focus for clean
attribution); batch-size + EMA tail experiments; final production window
values. See [[session-status-2026-07-06b]] for the detail.
