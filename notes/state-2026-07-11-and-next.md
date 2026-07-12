# State of the project (2026-07-11) and what must still be tested

The newest note: what happened in the 2026-07-11 sessions, exactly what
is unverified, and the ordered list of runs the user still owes the
code. A fresh session orients here first (via MEMORY.md), then reads
the topic notes it needs.

## Where the code stands

The queued program is CODE COMPLETE. One PyTorch training stack serves
five output families — cosmic shear (3x2pt data vectors), scalar
(named derived parameters), CMB spectra (TT/TE/EE/phi-phi), background
(H(z) + D_M), matter power (P_lin + the nonlinear boost) — each with
its geometry, loss, dataset generator, cobaya adapter, thin drivers,
and two acceptance gates. The board has 32 gates (count by enumerating
gates/board.py's registry, never by note arithmetic). Everything below
is on branch claude/amazing-keller-e798b6; the user merges to main and
pushes (only main is ever pushed; the GPU workstation pulls main).

## What landed on 2026-07-11 (the big day)

In order, with the commit that carries each:

1. **POL executed** (commits 6020349 / 7648db5 / 4629237): the main
   README consolidated (family sections 15-17, the Drivers table,
   appendix renumber; every claim census-verified), the EMUL2
   acceptance YAML shipped, the doc deep pass (AST-body-hash-proven
   doc-only over all 88 .py files), and the Alien-Python sweep
   (8 cold-path files, 7/7 old-vs-new output-equivalence probes).
2. **D-GEO5 — legacy geometry shims retired** (e622458): the user
   ruled no science artifact predates the geometries/ folder, so the
   six flat geometries_*.py shims were DELETED and the geo-paths gate
   inverted to "old paths are dead, loudly" (ModuleNotFoundError
   naming the path). A pre-GEO test artifact on the workstation will
   fail rebuild BY DESIGN — retrain it.
3. **D-MP8 — syren vendored** (e622458): the two symbolic_pofk modules
   the MPS base uses live in `syren/` (numpy-only; copied from the
   LEGACY emulmps bundle with the user's own edits — never PyPI;
   function bodies AST-identical; 5,433 values byte-identical;
   import-only deviations listed in syren/README.md). Nothing needs
   `pip install symbolic_pofk` anywhere anymore. Re-vendoring is a
   deliberate act + retrain.
4. **Per-family train drivers** (e622458): cosmic_shear_train_emulator
   gained `main(prog, family)` + `require_family_block` (a
   wrong-family YAML fails at startup NAMING the right driver); thin
   wrappers for cmb / baosn / mps.
5. **Family-first driver renames** (user ruling, supersedes the old
   verb-first namespace): `<family>_<verb>_emulator.py`, "what you are
   emulating comes first always", `single` dropped — 17 drivers + 5
   driver-named example YAMLs; run_board's `_DRIVER`, the board
   config's golden YAML (now cosmic_shear_train_emulator.yaml), the
   Optuna STUDY_NAME, and every reference swept LONGEST-FIRST
   (sweep_ntrain_scalar_emulator CONTAINS train_scalar_emulator — the
   substring trap).
6. **README section 18** (3b6724c): "Generating the training set"
   promoted from appendix to a numbered section, rewritten for all
   four generators + compute_cmb_covariance.py (it was lensing-only).
7. **MPS-DIAG closed** (4b0c480): the grid2d diagnostics pages. The
   key fact: in law space the residual pred - truth =
   ln(P_pred / P_truth) — THE SYREN BASE CANCELS — so the pages read
   directly as fractional error of the served spectrum. Two pages
   (median + worst |residual| heatmaps on a shared color scale; per-k
   bands at three redshifts); wired through plot_diagnostics(grid2d=),
   the train driver, and a new mps-smoke leg. Probe: the REAL function
   body exec'd on known answers, 4/4.
8. **D-CM12 + D-CM13 SPECS written, deliberately NOT implemented**
   (d6144ba): dense-Cinv CMB training off the non-Gaussian covariance
   blocks the covariance script already writes, and conv/TRF heads on
   the CMB path via a family-agnostic head_coords() geometry
   interface. Both await the user's audit and are sequenced AFTER the
   board baseline + EMUL2. Open ruling inside D-CM12: what the
   roughness term means under a rotated (non-diagonal) whitening.
9. **Family driver parity** (evening, user directive "mimic the
   cosmic-shear capabilities"): the cosmic-shear sweep_ntrain /
   sweep_hyperparam / tune drivers gained the train driver's
   main(prog, family) surface (+ out_default; per-family Optuna
   study names), and ALL twelve family sweep/tune drivers became
   thin wrappers over them — multi-GPU, --gpu-pack, LPT, and the
   journal study now work for every family; the earlier serial
   family_drivers loops were deleted (family_drivers.py keeps the
   sweep-block helpers, one definition tree-wide); gpu-pack's VRAM
   estimate handles the scalar family's dv-less data block. Pool
   workers proven AST-identical to HEAD; probes 4/4. Every thin
   wrapper (train included) now carries provenance comments naming
   where main lives and what the wrapper pins.
10. **The notes consolidation**: ~85 notes rewritten into
   ~11; the old files are retired but survive in git history — for any
   forensic question, `git log --follow` the old path.
11. **D-CM13 IMPLEMENTED, generalized (late evening, user order:
   "I want that for CMB and MPS minimum — I prefer that they all
   have", citing arXiv 2505.22574)**: the conv/TRF correction heads
   now ride cmb / grid / grid2d. The key simplification over the
   spec: the diagonal family geometries whiten IN physical order, so
   the heads' basis change degenerates to the identity (W_fd / W_df
   stay None; cosmic shear untouched — its geometry has evecs). The
   split is the new attach_head_coords() (cmb: one bin over ell;
   grid: one bin over z; grid2d: one bin per z slice), called at
   build_geometry AND inside results._rebuild_model so head
   artifacts rebuild from files alone. New knob model.trf.n_tokens
   re-segments a single-bin spectrum into attention windows (the
   paper's tokenization). Scalar stays trunk-only (no coordinate
   axis). Head legs added inside cmb-identity + mps-identity (board
   count unchanged at 32). Also proven the same evening: trim /
   focus / berhu ladder + anneals / EMA / clip / rewind / optimizer-
   lr-scheduler blocks were ALREADY family-universal (the shared
   loop; the family example YAMLs now advertise the optional guards
   as commented blocks). Discovered in passing: cosmic-shear head
   artifacts could not rebuild (bin_sizes never persisted nor
   re-attached on the rebuild path) — fixed in item 12. Full record:
   families-scalar-cmb.md (the D-CM13 section).
12. **The cosmic-shear head-artifact rebuild fix** (same evening):
   DataVectorGeometry.state() now persists bin_sizes (+ pm_kept)
   when build_shear_angle_map attached them — schema-additive, the
   section_sizes/probe pattern; __init__ gained the optional kwargs
   with the attribute-UNSET-when-None rule (the hasattr guards in
   ResCNN / ResTRF / BlockDiagonalGeometry keep working); a
   pre-persistence head file is refused loudly at rebuild ("bin-split
   persistence"), never re-derived (rebuild must not need ROOTDIR
   data files). save-rebuild-drift gained a rescnn head variant
   (real training path, bitwise round-trip proving the whole
   attach -> save -> rebuild chain) + a deleted-split refusal leg.
   Trunk-only artifacts' state is byte-identical to before.

## Evidence status: what is PROVEN vs what is PENDING

Mac-proven (this machine has numpy only — no torch/h5py/yaml/scipy):
every unit passed compileall + AST censuses + numeric probes (numpy
mirrors, or better: the real function body exec'd under tensor-like
fakes on known answers), both README censuses (anchors + paths), the
board-registry census (32), and the stale-reference censuses. The
syren vendoring and the background/MPS pure math executed FOR REAL
here (numpy-only paths).

PENDING — nothing torch-side has run since the board was 25/25:

## The tests the user still must run (in this order)

1. **Merge + push main** (the branch is ahead; merges are clean now —
   the README-conflict era ended when main took the consolidated
   README).
2. **The 32-gate board on the workstation**: `git pull`, drop any
   local board-config override (`git checkout --
   gates/board_config.json` — the golden config name changed in the
   rename), then the full board. New/changed since the last green:
   cmb-identity (now incl. the D-CM13 ResTRF head leg), cmb-smoke
   (SLOW: ~400 serial CAMB calls), bsn-identity, bsn-smoke,
   mps-identity (now incl. the D-CM13 ResCNN head leg), mps-smoke
   (includes the MPS-DIAG pages leg), geo-paths (inverted: old paths
   must be DEAD), plus `--force-rerun scalar-identity scalar-smoke
   save-rebuild-drift` (all three were green on the 25/25 board and
   gained legs since — save-rebuild-drift now carries the rescnn
   head round-trip + the pre-persistence refusal). Full green is
   simultaneously GEO's acceptance and the board baseline for
   everything after.
3. **Train the five production artifacts** via the family drivers:
   rdrag (scalar), hubble + dm (baosn), pklin + boost (mps).
4. **The EMUL2 acceptance**: cobaya_theory/EXAMPLE_EMUL2_EVALUATE.yaml
   — cosmolike `use_emulator: 2` consuming emul_mps + emul_baosn +
   emul_scalars; fill the artifact path roots (placeholders under
   projects/lsst_y1/emulators/); nothing to pip-install. Recorded
   first-run risks: the Pk_grid requirement resolution onto a
   non-CAMB theory, and the generator's Pk_interpolator requirement
   against the workstation's cobaya version — both self-diagnosing.
5. **Audit the D-CM12 / D-CM13 specs** (in families-scalar-cmb.md)
   and rule on roughness-under-rotation.
6. **Then the science thread** (all workstation): D-TP9
   frozen-vs-refine at fixed N_train, berhu attribution, bs + EMA,
   the activation bake-off, NPCE at scale; the EDE + w(z)-PCA
   transfer program and the first real LCDM -> w0waCDM fine-tune wait
   on real extended-model training dumps.

## Standing constraints that gate future work

- Transfer learning is PERMANENTLY exclusive to the cosmolike + CMB
  data-vector families; fine-tuning is universal.
- The BSN family is flat-only V1 (the legacy curvature formula was
  dimensionally wrong and is deliberately not reproduced; a curvature
  branch would be a new spec).
- The board sequencing rule: no new design surface before the first
  full 32-gate green + the EMUL2 acceptance.
- Every YAML change shown as a paste-ready block; terminal output
  essential-only; artifacts self-describing (never-trust-defaults);
  C-readable Python on cold paths; smoke gates must fail on a dead
  network. The full rule set: conventions-and-workflow.md.
