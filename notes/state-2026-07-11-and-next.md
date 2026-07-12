# State of the project (through 2026-07-12) and what comes next

The read-first note: where the code stands, what is proven, and the
ordered list of runs the user still owes the code. A fresh session
orients here first (via MEMORY.md), then reads the topic notes it
needs. The 2026-07-11/12 events lived here as a 21-item ledger while
they unfolded; the durable knowledge has been REDISTRIBUTED to the
topic notes (the map below), and the full chronology survives in git
history (`git log -p notes/state-2026-07-11-and-next.md`).

## Where the code stands

The queued program is CODE COMPLETE. **The board's first full 32/32
green was run 9 (2026-07-12 00:12)** — simultaneously GEO's
acceptance and the board baseline for D-CM12 and the science thread;
the CURRENT standing is run 11's 30/32 (one red fixed on the branch
awaiting rerun, one open — the evidence section below). One PyTorch training stack serves five
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
ruling "any trunk-head design could benefit" — the NPCE closed-form
base on every family (the same day's second ruling: pce: fits a
sparse-Legendre trunk under any refiner, residual-only off cosmic
shear, cmb only with amplitude_law none), and frozen-base transfer
on every family but scalar (the overnight third ruling overturning
the BAOSN/MPS permanent forbids and closing D-CM7; whitened space,
sum recommended, refine cosmolike-only)). Count gates by enumerating
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
(2026-07-12); run 10 (the `--force-rerun-all` regression pass at that
same HEAD) re-proved the cosmic-shear era, all four smokes, the head
legs, and the constant pins at 29/32, its three harness reds fixed and
then PROVEN green by run 11 — the post-merge pass at merged HEAD
4e783fa (30/32), which also first-executed the NPCE check_npce legs
GREEN in all four family identity gates and the two-phase
phase-discipline legs GREEN. Run 11's two reds are both
first-executions of overnight legs: cmb-smoke's eq-6 leg (the clpp
re-lensing array was truncated below CAMB's required Params.max_l
length — root-caused from CAMB's own source, fixed on the branch, and
the leg's failure detail now carries the stdout tail cobaya's
exception hook logs to) and transfer-identity's check_diagonal (OPEN —
awaiting gates/logs/transfer-identity.log; the diagonal-transfer
algebra is Mac-probe-proven, so the suspect surface is the real-torch
save -> rebuild -> predict roundtrip). The full run-by-run record:
gates-and-board.md.

## The runs the user still owes the code (in this order)

1. **The D-CM11-A unit, then one rerun** (runs 10 + 11 retired the
   full passes): the Implementer lands the eq-6 normalization fix
   (the unit queue below), the user pastes
   gates/logs/transfer-identity.log for the open red, then one
   workstation pass:
   `python gates/run_board.py --force-rerun cmb-identity cmb-smoke transfer-identity`
   (~12 min; any covariance failure now names its own cause via the
   stdout tail).
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

## The implementer unit queue (red-team static audit, 2026-07-12)

The audit record is commit 1fc90c8 on codex/architect-docs-static-audit
(NEVER merge that branch — its notes conflict with the consolidated
ones; cite it by SHA). One unit per handoff, each with its own green
evidence and notes update. The Architect verifies and specs each unit
at its own handoff time — unit 1 is verified and specced; 2-5 are
accepted audit CLAIMS awaiting their own verification:

1. D-CM11-A, the eq-6 normalization (spec of record:
   families-scalar-cmb.md "D-CM11-A") — HANDED OFF 2026-07-12.
2. Dataset readiness (a failed generator row must fail the
   generation, and staging must reject flagged rows), then the MPS
   sigma8 h-unit consistency.
3. Best-record truth: one authoritative best record including epoch
   0, agreed by trainer, drivers, tuner objective, and artifact
   attrs.
4. Harness/CLI truth: bad selectors and unknown flags exit nonzero
   before any work; preflight covers the executed surface.
5. Small contracts: the scalar-trainer contract; NPCE zero-kept-mode
   honesty; then the lower-priority hardening ledger (asserts ->
   typed errors on config surfaces, covariance/axis validation,
   device-string strictness, the torch.load trust boundary, plot
   color mapping) in separate commits.

## Standing constraints that gate future work

- Transfer learning rides cosmolike + cmb + grid + grid2d since the
  2026-07-12 overnight symmetry ruling (the user overturned the
  BAOSN/MPS permanent forbid and closed D-CM7); SCALAR stays out
  (D-SP8, a recorded ruling the user may overturn). Fine-tuning is
  universal; heads-on-scalar is a permanent no (no coordinate axis
  between named outputs).
- The BSN family is flat-only V1 (the legacy curvature formula was
  dimensionally wrong and is deliberately not reproduced; a curvature
  branch would be a new spec).
- The sequencing rule: no new design surface before the EMUL2
  acceptance (the board-green half is now met).
- Every YAML change shown as a paste-ready block; terminal output
  essential-only; artifacts self-describing (never-trust-defaults);
  C-readable Python on cold paths; smoke gates must fail on a dead
  network. The full rule set: conventions-and-workflow.md.
