# State of the project (through 2026-07-12) and what comes next

The read-first note: where the code stands, what is proven, and the
ordered list of runs the user still owes the code. A fresh session
orients here first (via MEMORY.md), then reads the topic notes it
needs. The 2026-07-11/12 events lived here as a 21-item ledger while
they unfolded; the durable knowledge has been REDISTRIBUTED to the
topic notes (the map below), and the full chronology survives in git
history (`git log -p notes/state-2026-07-11-and-next.md`).

## Where the code stands

The queued program is CODE COMPLETE and **the 32-gate board is
GREEN — the first full 32/32, board run 9, 2026-07-12 00:12**. That
green is simultaneously GEO's acceptance and the board baseline for
D-CM12 and the science thread. One PyTorch training stack serves five
output families — cosmic shear (3x2pt data vectors), scalar (named
derived parameters), CMB spectra (TT/TE/EE/phi-phi), background
(H(z) + D_M), matter power (P_lin + the nonlinear boost) — each with
its geometry, loss, dataset generator, cobaya adapter, thin drivers,
and two acceptance gates, and with FULL capability symmetry: every
family has the whole training surface (loss ladder + anneals,
trim/focus, EMA, clip/rewind, fine-tuning, multi-GPU drivers, the
conv/TRF correction heads everywhere a coordinate axis exists, the
two-phase trunk-then-head schedule on every head model with the
trunk:/head: blocks and the per-head activation pin — the 2026-07-12
ruling "any trunk-head design could benefit" — and, since the same
day's second ruling ("nothing should prevent PCE as the trunk"), the
NPCE closed-form base on every family: pce: fits a sparse-Legendre
trunk under any refiner, residual-only off cosmic shear, cmb only
with amplitude_law none). Count gates by enumerating
gates/board.py's registry, never note arithmetic. The user merges to main and pushes (only main is ever
pushed; the GPU workstation pulls main).

## Where the 2026-07-11/12 knowledge now lives (the redistribution map)

- POL, D-GEO5 shim retirement, the notes consolidation, the
  family-first rename, the capability-symmetry + board-saga arc:
  project-and-history.md (arc items 9-10; the recipe's driver step
  is updated to the wrapper architecture).
- The driver surface (family-first names, thin wrappers over
  main(prog, family), full multi-GPU parity, the sweep-block
  helpers): training-stack.md ("The driver surface").
- D-CM13 implemented + generalized (heads on cmb/grid/grid2d,
  identity basis, attach_head_coords, n_tokens, the scalar
  exclusion) and the cosmic-shear head-artifact rebuild fix
  (bin_sizes/pm_kept persistence): families-scalar-cmb.md (the
  D-CM13 section), models-and-designs.md, and
  artifacts-inference-warmstart.md (the rebuild-side facts).
- D-CM12: still a SPEC AWAITING AUDIT — families-scalar-cmb.md.
- The stale-background saga (the falsified wants-Cl-quirk
  hypothesis, the logposterior(cached=False) lifecycle, the
  dump-variance tripwire) and D-MP9 (constant-column pinning, born
  then amended law-agnostic) and the grid-derived k_max:
  families-background-mps.md; the two-evaluation-idioms rule and the
  covariance-script fixture conventions: data-generation-and-cuts.md.
- The nine board runs, every red root-caused, run by run + the
  --force-rerun-all flag: gates-and-board.md (the run-history table
  + notable facts).
- The new recurring process lessons (HEAD-at-run before reading a
  red; fixtures mirror shipped examples; derive coupled fixture
  widths; $ROOTDIR extends to in-process get_model; carve guards on
  physics axes; falsifiable tripwires before trusting a fix):
  conventions-and-workflow.md ("Process lessons that recur").
- Family acceptance stamps: SPE closed (07-10); CME accepted end to
  end (run 4); BSN accepted (run 6); MPS accepted (run 9, rel 0.93%
  vs CAMB against the 5% bar) — each family note's section header.

## Evidence status

Everything through board run 9 is workstation-proven at HEAD 4c65331
(2026-07-12). Still NEVER EXECUTED, by resume accounting: the D-CM13
head legs inside cmb-identity / mps-identity, the D-MP9 legs in
mps-identity, the two-phase phase-discipline legs in the same head
checks, and the NPCE check_npce legs in all four family identity
gates (those gates' greens date from run 1, before the legs
existed). The `--force-rerun-all` regression pass is therefore both
the cosmic-shear no-regression proof and those legs' first execution.

## The runs the user still owes the code (in this order)

1. **The full regression pass** (workstation):
   `python gates/run_board.py --force-rerun-all` — re-executes every
   gate including the twelve cosmic-shear-era ones (with
   ema-off-identity's golden byte-identity leg vs the pre-EMA
   commit) AND first-executes the head/D-MP9 identity legs. Budget
   ~1 h (cmb-smoke's ~400 CAMB calls dominate; mps-smoke is ~12 min
   since the k_max derivation).
2. **Train the five production artifacts** via the family drivers:
   rdrag (scalar), hubble + dm (baosn), pklin + boost (mps).
3. **The EMUL2 acceptance**: cobaya_theory/EXAMPLE_EMUL2_EVALUATE.yaml
   — cosmolike `use_emulator: 2` consuming emul_mps + emul_baosn +
   emul_scalars; fill the artifact path roots (placeholders under
   projects/lsst_y1/emulators/); nothing to pip-install. The two
   recorded first-run risks are now KNOWN quantities from the board
   saga: cobaya's >= 4-redshift Pk_interpolator constraint and the
   $ROOTDIR-relative theory paths.
4. **Audit the D-CM12 spec** (families-scalar-cmb.md) and rule on
   roughness-under-rotation; NB a dense CMB geometry would carry an
   eigenbasis — revisit the D-CM13 identity shortcut there.
5. **The science thread** (all workstation): D-TP9 frozen-vs-refine
   at fixed N_train, berhu attribution, bs + EMA, the activation
   bake-off, NPCE at scale, the head bake-offs (restrf + n_tokens on
   CMB is arXiv 2505.22574's headline case); the EDE + w(z)-PCA
   transfer program and the first real LCDM -> w0waCDM fine-tune
   wait on real extended-model training dumps.

## Standing constraints that gate future work

- Transfer learning is PERMANENTLY exclusive to the cosmolike + CMB
  data-vector families; fine-tuning is universal; heads-on-scalar is
  a permanent no (no coordinate axis between named outputs).
- The BSN family is flat-only V1 (the legacy curvature formula was
  dimensionally wrong and is deliberately not reproduced; a curvature
  branch would be a new spec).
- The sequencing rule: no new design surface before the EMUL2
  acceptance (the board-green half is now met).
- Every YAML change shown as a paste-ready block; terminal output
  essential-only; artifacts self-describing (never-trust-defaults);
  C-readable Python on cold paths; smoke gates must fail on a dead
  network. The full rule set: conventions-and-workflow.md.
