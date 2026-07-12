# State of the project (through 2026-07-12) and what comes next

The read-first note: where the code stands, what is proven, and the
ordered list of runs the user still owes the code. A fresh session
orients here first (via MEMORY.md), then reads the topic notes it
needs. The 2026-07-11/12 events lived here as a 21-item ledger while
they unfolded; the durable knowledge has been REDISTRIBUTED to the
topic notes (the map below), and the full chronology survives in git
history (`git log -p notes/state-2026-07-11-and-next.md`).

## Where the code stands

The feature program reached code-complete, but the 2026-07-12 red-team
static review found release blockers outside the board's current
gate suite: production grid2d staging defeats the memory ladder; standard
generated `.1.txt` files bypass the parameter-order sidecar check; dump
members and artifact pairs are not identity-bound; and parallel worker
success is not truthfully accounted. Treat "code complete" below as
feature coverage, not a release-readiness claim, until the open audit
queue closes. **The board's first full 32/32
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

1. **Close the two remaining workstation reds.** The eq-6 normalization
   implementation is already merged at HEAD 7f455e6 and independently
   accepted on the Mac math scope (maximum relative error about 8.1e-14).
   Its gate fixture still needs the recorded raw-vs-CAMB-scaled
   lens-potential delta. The user also pastes
   gates/logs/transfer-identity.log for the open diagonal-transfer red,
   then runs one workstation pass:
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
evidence and notes update. ALL FIVE UNITS VERIFIED against the code
by the Architect (2026-07-12) — every audit claim reproduced; the
evidence line under each unit is the anchor a spec starts from:

1. D-CM11-A, the eq-6 normalization (spec of record:
   families-scalar-cmb.md "D-CM11-A") — IMPLEMENTED (Opus) and
   independently Architect-audited ACCEPTED on Mac scope at merged HEAD
   7f455e6; the family note's later provenance-correction section is the
   real audit record (the earlier pre-written Fable verdict and commit
   attribution are invalid). One raw-vs-scaled known-answer fixture
   delta plus the
   workstation pass remain before close.
2. Dataset readiness + MPS sigma8. VERIFIED: run_generator ends
   MPI.Finalize(); exit(0) unconditionally (generator_core.py tail)
   while per-sample failures zero the dv row and mark the failfile;
   emulator/data_staging.py contains ZERO failfile references — the
   trainer can stage fabricated rows. _compute_sigma8 (emul_mps.py)
   integrates k [1/Mpc] with R = 8.0 — that is sigma(8 Mpc), not
   sigma8's 8 Mpc/h; its own docstring flags the legacy convention.
   CAVEAT for the handoff: the sigma8 fix changes legacy-served
   values — needs the user's ruling (the BSN-curvature precedent:
   dimensionally wrong legacy math is not reproduced). Also verified
   for this unit: --boundary values outside (0, 1) are silently
   rewritten to 1 (generator_core.py ~231) instead of rejected.
3. Best-record truth. VERIFIED: training.py seeds best-tracking with
   the epoch-0 baseline and restores those weights, but the driver
   (cosmic_shear_train_emulator.py ~350: "fracs[i][0] is frac>0.2 at
   epoch i+1") and the tuner objective both recompute "best" over a
   history that STARTS AT EPOCH 1 — on a no-improvement warm start
   the restored model, console line, h5 attrs, and Optuna objective
   disagree.
4. Harness/CLI truth. VERIFIED: run_board.py prints "warning:
   unknown gate id" and PROCEEDS (exit 0); --force-rerun ids are
   never validated (a typo is silently ignored while the resume
   prints a green summary); the generator CLI parse_known_args
   discards unknown flags (the shared drivers repeat the pattern);
   and the preflight dirty-tree watch covers emulator/, gates/, and
   the root *.py only (run_board.py ~615) — compute_data_vectors/,
   cobaya_theory/, and syren/ are unwatched.
5. Small contracts. VERIFIED: (a) the shared driver reads
   cfg["data"]["train_dv"] unconditionally (~214 run_tag, ~379
   attrs) while the scalar data block documents that key as
   forbidden — the scalar DRIVER path has never executed end to end
   and blocks the queued rdrag artifact training (SEQUENCING: this
   fix must land before the five-artifacts step); (b)
   designs/pce.py:414 "if not cols: # always keep mode 0" —
   verbatim the audit's zero-kept dishonesty (a mode that failed the
   LOO gate is kept silently); (c) hardening-ledger instances
   confirmed: parameter.py:256 validates user samples via assert
   (python -O strips it), results.py:614 torch.load without
   weights_only, _pick_device returns cpu for an unrecognized
   device string, plot_xi carries mutable list defaults.

### Second red-team wave: verified open queue

These findings are recorded in their existing topic notes; no new note
files were created. Priority follows user-visible risk:

1. **Bounded grid2d staging (next handoff).** The shipped 50,000-row,
   122 x 2,000 MPS setup selects/casts whole unthinned raw and base
   matrices before `k_stride`; at least two 90.897 GiB float64
   selections coexist. Spec and gate: data-generation-and-cuts.md,
   "Grid2d staging defeats its own memory ladder". This closes the
   ordinary/finetune/frozen-transfer paths; production-scale grid2d PCE
   remains a separately recorded low-rank-fit design problem.
   VERIFIED (Fable, 2026-07-12): experiment.py _grid2d_law_rows —
   `dv_rows = np.asarray(src["dv"][rows_sorted], dtype="float64")` at
   the full unthinned width, a same-width base_rows copy, the log
   quotient as a third array, and only THEN the k_stride column cut;
   all after load_source's RAM/memmap decision, which it defeats.
2. **Data-selection truth.** The no-cut scalar/CMB/grid/grid2d
   learning-curve wrappers fail in `pool_size` by indexing an absent
   `omegabh2_hi`; ordinary `load_source` looks for
   `X.1.paramnames` while the generator writes `X.paramnames`, disabling
   its own order guard; same-shaped params/dv/base files can be mixed
   silently. Spec: data-generation-and-cuts.md.
3. **Artifact-pair integrity.** Direct two-file overwrite plus no digest
   lets same-shaped wrong weights load strictly against an unrelated h5.
   Syren-law artifacts also do not bind the inference formula version.
   Spec: artifacts-inference-warmstart.md.
4. **Parallel truth and cleanup.** Parallel Optuna ignores worker exit
   codes and accepts any historical COMPLETE trial; the shared GPU pool
   can orphan children or wait forever on invalid token plans. Spec:
   training-stack.md.
5. **Live resource sizing and result-table truth.** `dv_len=3000` is
   hardcoded across dense and wide-diagonal losses; sweep-table length
   mismatch truncates silently. Spec: training-stack.md.
6. **Python documentation truth.** The exact census and remaining
   internal-ledger prose are in conventions-and-workflow.md. This is a
   separate doc-only unit after correctness work, proven by an
   AST-minus-docstrings hash.
7. **Real Cobaya adapter contract.** The MPS adapter advertises outputs
   through the input-support hook; scalar calculate does not create the
   required derived-state mapping; the MPS amplitude alternative is
   narrowed to `As` despite the shared reader accepting `As_1e9`.
   Stubbed identity gates miss all three. Spec:
   artifacts-inference-warmstart.md.
   VERIFIED (Fable, 2026-07-12): emul_mps.py get_can_support_params
   returns ['Pk_grid', 'Pk_interpolator', 'sigma8'] (the input-param
   hook; products belong in get_can_provide / must_provide);
   emul_scalars.py calculate writes derived only under
   `if want_derived and "derived" in state` (real cobaya never
   pre-creates the key — the theory must); emul_mps.py hard-codes
   req["As"] while syren_params_from (syren_base.py) accepts
   As_1e9 | As, and the training dumps themselves sample As_1e9
   (analytics.py reads that column by name).
8. **Checkpoint-set integrity.** Axis sidecars and `.paramnames` are not
   in the resume census, multi-file append is not one transaction, and a
   load error falls through to fresh generation on the same roots — an
   intended append can replace the prior set. Spec:
   data-generation-and-cuts.md.
   VERIFIED (Fable, 2026-07-12): generator_core.py __load_chk census =
   dv members + fail file + covmat + ranges + X.1.txt, no .paramnames
   and no axis sidecars; __save_chk publishes per file (each tmp +
   os.replace atomic, the SET sequential); __run_mcmc wraps __load_chk
   in `except Exception` -> loadedfromchk = False -> the fresh-run
   branch on the SAME roots — the very row-count ValueError that means
   "this set is inconsistent, stop" is converted into a replacement.
9. **Validation and diagnostic memory truth.** Validation ignores its own
   safe chunk and uses the train chunk; the generic local-linear floor
   expands as N_val x 40 x output width and is not runnable on production
   MPS. Spec: training-stack.md.
   VERIFIED (Fable, 2026-07-12): build_loaders (batching.py) computes
   and returns data["val"]["load"] planned against budget - used_tr;
   training.py sets `load = data["train"]["load"]` and passes THAT to
   every eval_val call; data["val"]["load"] is never read anywhere.
   local_linear_floor (diagnostics.py) stages both whole sources on
   the device and materializes Ttr[nbr] = (N_val, k_nn=40, out_dim).
10. **Activation-bakeoff liveness.** Its bespoke parent blocks on a fixed
    count of un-timed queue reads before joining children; worker failures
    during setup/staging/geometry emit no result and hang the command. Spec:
    training-stack.md.
    VERIFIED (Fable, 2026-07-12): the worker's try/except covers ONLY
    the per-activation train/frac block; from_config, stage_val,
    stage_train, and build_geometry sit outside it, and the parent
    loops `for _ in range(total): result_q.get()` with no timeout
    before any join. The worker docstring's "never deadlocks" promise
    is false for every pre-training failure.
11. **Geometry numerical and read-side integrity.** Covariance builders take
    square roots without an SPD check, block whitening clips singular modes
    to zero and divides by them, and saved geometry states are rebuilt without
    finite/shape/invertibility validation. Spec:
    artifacts-inference-warmstart.md.
    VERIFIED (Fable, 2026-07-12); the Fable "no clip exists anywhere"
    correction was RETRACTED the same day (a truncated verification
    grep — `head -20` cut the hits; the red team re-checked and was
    right). TWO mechanisms coexist in emulator/geometries/output.py
    and BOTH are recorded: (a) from_cosmolike runs eigh then
    np.sqrt(lam) UNCHECKED — a numerically negative eigenvalue becomes
    NaN silently, an exact zero gives sqrt_ev = 0, and whiten divides
    with no floor; (b) BlockDiagonalGeometry._build_block runs
    np.sqrt(np.clip(lam, 0.0, None)) — a negative eigenvalue is
    clipped to a ZERO divisor for its whiten, exactly the original
    red-team wording. from_state is `cls(device, **state)` with no
    finite/shape/invertibility validation on anything it loads. The
    fix contract stands unchanged.
12. **Optimized-mode validation parity.** Seventeen runtime guards across
    batching, model designs, geometry, and loss code are `assert` statements
    and disappear under `python -O`. Spec: conventions-and-workflow.md.
13. **Covariance-input validation** (red-team, 2026-07-12 third wave).
    cov_args has no schema/range boundary: band_width <= 0 hangs
    band_windows; step_fracs order silently changes the kept
    derivative (the code keeps stack[0], the docs promise the
    smallest step); lens_lmax beyond the supplied raw power is
    silently zero-padded and the zero-band guard then deletes real
    eq-6 contributions; fsky is unvalidated (0 -> inf, negative ->
    NaN) and silently defaults; no key whitelist and a quoted
    "false" enables the non-Gaussian path. Spec (with the required
    contract + gate legs): families-scalar-cmb.md,
    "Covariance-input validation unit".
    VERIFIED (Fable, 2026-07-12): all six failure paths re-derived
    against the code — band_windows ~279-285 (start never advances at
    width 0), stack[0] at ~580 vs the ~545 comment, the zero-pad at
    ~529-531 interacting with the D-CM11-A zero-band guard (data
    absence masquerades as physical zero; the width-1 arithmetic
    [0.4, 2/7, 2/9] confirms the deleted weights), the pp request
    capped at lmax (~415), fsky/.get defaults (~686, ~517-519,
    ~706), and bool(str) truthiness at ~728/759. Queued AFTER the
    in-flight transfer-fixture unit and the grid2d staging unit.
14. **Finite training/evaluation contract** (red-team fourth wave,
    2026-07-12, CRITICAL — JUMPS THE QUEUE to right after the
    in-flight fixture unit: it protects every production training,
    and the five artifacts are next). NaN chi2 scores as a perfect
    emulator: frac counts NaN below every threshold, the best-epoch
    rule snapshots NaN weights, no finite guard on loss/grad/step,
    and the dead-network smoke bars PASS a NaN run. Spec:
    training-stack.md, "NaN scores as a perfect emulator".
    VERIFIED (Fable, 2026-07-12): training.py ~1427-1433 (median
    propagates NaN, (c > t) counts NaN below), ~2047 (f0 = 0.0 wins),
    ~1971-1980 (backward/clip/step unguarded), zero isfinite/isnan
    in the file.
15. **BAOSN physical-domain + pair-shape guards** (fourth wave).
    Zero/negative H accepted (NaN/negative distances served);
    z-pairs unvalidated (reversed pair -> negative D_A, silent
    3-col, IndexError 1-col). Spec: families-background-mps.md.
    Land BEFORE the EMUL2 acceptance.
    VERIFIED (Fable, 2026-07-12): background.py has no finite/
    positivity/monotonicity guard; emul_baosn ~356-368 documents
    z1 <= z2 but enforces nothing.
16. **MPS query/composition totality** (fourth wave). check_ranges
    passes NaN (comparisons only) and empty queries hit builtin
    min(); NaN extrap bounds stored then defeat the range check;
    pk_nl = boost * pk_lin never validated (finite factors can
    overflow to a cached Inf). Spec: families-background-mps.md.
    Land BEFORE the EMUL2 acceptance.
    VERIFIED (Fable, 2026-07-12): emul_mps ~144-164 (guards are
    </> only), ~94-114 (NaN bound kept when the extend-check branch
    is skipped).
17. **Generator ingress identity** (fourth wave). ord validated by
    set equality only — duplicates pass and the two reorder helpers
    collapse them DIFFERENTLY (sampler dim 4 vs model dim 3 in the
    counterexample); covmat-header pidx keeps the last duplicate;
    thin/unique shortfall publishes a smaller dataset with a
    warning; unparsed CLI args accepted. Spec:
    data-generation-and-cuts.md, "Generator ingress identity".
    Land before any new production dataset generation.
    VERIFIED (Fable, 2026-07-12): generator_core ~337 (set
    equality), ~458-466 (the two helpers collapse differently by
    inspection), ~326 (last-dup pidx).
    VERIFIED (Fable, 2026-07-12) as a class; the census at this HEAD is
    EIGHTEEN `^assert` statements (batching 1, designs/ia 6,
    designs/plain 6, designs/blocks 1, losses/core 1,
    geometries/output 2, geometries/parameter 1), all config/geometry/
    user-data guards. The red team concurs (2026-07-12): seventeen was
    a counting error on their side; eighteen stands at 4f4dab3 and at
    the reviewed HEAD.

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
