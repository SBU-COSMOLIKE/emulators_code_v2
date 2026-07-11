# The project and how it was built

Consolidated 2026-07-11 from ~85 topic notes (retired; every old note
survives in git history — `git log --follow notes/<old-name>.md`).
Read notes/MEMORY.md first, then state-2026-07-11-and-next.md for
where things stand.

## What this is

A PyTorch emulator program for cosmological inference inside Cocoa:
networks that replace expensive physics computations (cosmolike 3x2pt
data vectors, CAMB CMB spectra, the background, the matter power
spectrum, derived scalars) inside cobaya MCMCs. The goal metric is
SAMPLE EFFICIENCY: f(delta-chi2 > 0.2) over validation cosmologies vs
N_train; the target regime is high temperature + w0wa + TATT, where
training cosmologies are the expensive object. One training stack
serves five output families; artifacts are self-describing (the
never-trust-defaults doctrine); every feature lands with acceptance
gates the user runs on a GPU workstation.

## The development arc (commit archaeology by phase)

1. **The notebook era (June 2026).** Everything began as the teaching
   notebook pytorch1.ipynb (read-only reference). Early experiments
   established the science doctrine and the CLOSED-experiment ledger
   (models-and-designs.md): the floor is data/coverage, factoring
   beats sampling, the correction-head idea.
2. **The package translation (late June).** Byte-faithful extraction
   into emulator/ + drivers; the cocoa layout (--root/--fileroot/
   --yaml, ROOTDIR-relative); EmulatorExperiment as the one setup
   object; the multi-GPU sweep machinery; the [default, min, max,
   kind] search convention.
3. **The training-stack build-out (2026-07-01..07).** Two-phase
   trunk/head training with zero-init identity heads; trim/focus;
   berhu losses + anneal; EMA with the snapshot-coupling invariant;
   phase blocks + single-phase demotion; absolute n_train/n_val;
   derived eval batch; the weight-decay allowlist; nested model
   config; per-head activations; model.norm; NPCE wiring; factored
   IA (NLA shipped, TATT live). Two whole-tree audits (style, then
   documentation truth) forged the house rules; the README campaign
   produced the two-README split and the didactic standard.
4. **The board (2026-07-07..08).** The self-driving gate harness the
   user runs; ~19 gates grown by every unit after; ten runs peeled
   eight wiring layers with ZERO physics bugs (gates-and-board.md
   carries the run table and lessons).
5. **Save schema v2 + adapters (2026-07-07..08).** Resolved-values
   artifacts, rebuild_emulator, EmulatorPredictor, the thin
   emul_cosmic_shear adapter; GSV/GCT gates; the dv_return design.
6. **Family folders (GRF, 2026-07-08) + portable board config (GBC,
   07-09).** designs/ + losses/; the refactor proven
   output-transparent by a fresh green board.
7. **Warm starts (2026-07-09..10).** FTW fine-tuning (block-extended
   geometry, epoch-0 parity) then TPE transfer learning (frozen base
   + parallel correction, embed-the-base artifacts, refine + the
   shared L2-SP anchor) — the infrastructure capstone: four training
   modes with one dial. Transfer later ruled EXCLUSIVE to the
   cosmolike + CMB data-vector families, permanently.
8. **The five-family program (2026-07-10..11).** SPE scalars (closed
   25/25, five board runs, the lesson bank); then in one long
   Architect-implemented window: CME (CMB spectra: the covinv ruling,
   amplitude law, covariance script, roughness loss, family
   diagnostics dispatch), BSN (two-regime background), MPS
   (correction-to-syren with the D-MP2-A base-on-disk flow, EMUL2),
   GEO (the geometries/ folder; shims later retired by D-GEO5), and
   POL (README consolidation + doc pass + the Alien-Python sweep).
   Board 23 -> 32 (count by enumerating the registry — a +1
   note-arithmetic error survived days once).
9. **The 2026-07-11 wrap** (state-2026-07-11-and-next.md): syren
   vendored in-repo; per-family train drivers + the family-first
   rename (<family>_<verb>_emulator.py, user ruling); README section
   18; MPS-DIAG; the D-CM12/D-CM13 specs; this notes consolidation.

## The family-pattern recipe (what a NEW output family adds)

SPE established it; CME/BSN/MPS instantiated it; the code map carries
it as a change-X row. A new family adds: (1) a data.<family> block
key, mutually exclusive with all others; (2) a pure validator
(rescale/ia/pce forbidden; transfer per the scope ruling; finetune
admitted with a pin); (3) a from_config branch — param_cuts optional,
trunk-only head guard via model_cls.head_block, the finetune sub-path
validated BEFORE any model-block read, and the cosmolike finetune
branch guarded `not self._<family>` (the ordering hazard, hit three
times); (4) a geometry in emulator/geometries/ persisting RESOLVED
values with state() byte round-trip and the relative zero-variance
guard; (5) a loss exposing the per-sample chi2 interface so
trim/focus/berhu/EMA/anchor compose; (6) staging by NAME through the
.paramnames sidecar (chain-root-aware); (7) results.py info flag +
CLASS-GUARDED metadata reads (the two-registry law collision);
(8) a predictor branch with a family-shaped return; (9) a cobaya
adapter on the emul_scalars template with wrong-kind guards BOTH ways;
(10) a thin <family>_train_emulator.py wrapper + the serial
sweep/tune pair on family_drivers; (11) TWO board gates —
<family>-identity (bitwise round-trips; every closed delta gets a leg
forever) and <family>-smoke (real generator -> train -> real cobaya
lifecycle; dead-network-RELATIVE bars) + registry registration;
(12) an example YAML carrying ALL required train_args blocks
(subscript-census-validated); (13) diagnostics pages behind the
family dispatch; (14) a README section (define-or-drop, snippets,
pipeline diagram) + code-map rows; (15) a generation driver on
generator_core.

## Program-level lessons (the ones that repaid their cost)

- The note is the spec of record; paraphrase is not a source (read
  legacy code before porting; port nothing except declared verbatim
  numerics, bug-for-bug, with findings recorded).
- Bitwise/byte-identity is the acceptance currency — and it is
  demanded only on same-computation legs; cross-path legs relax to a
  documented ~1e-6 (ruled three times).
- Structural fixes beat behavioral patches (the D-B1 arc: a patch
  introduced by a flawed schema was deleted, not adapted).
- A smoke that cannot fail a dead network proves nothing; a check
  harness is verified against a known-good case before its verdict
  counts; honest margins are recorded, not rounded.
- Fixing layer N unmasks layer N+1 — triage reds by layer (harness /
  check / config / library / contract); the whole board history found
  one library bug and zero physics bugs.
- Wide sweeps need count-asserted, truncation-free censuses (a head -5
  once hid rename stragglers; substring collisions require
  longest-first replacement).
