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
   EXTENDED (45M-67, sixteenth batch): sigma8's domain holes are
   in-unit — the nearest-z-within-0.01 branch relabels P(k, 0.009)
   as z = 0 (0.896% deterministic bias on a toy growth law) and
   the top-hat integral runs on whatever k interval the validator
   admitted (lo < hi, nk >= 8 is the WHOLE constraint): the
   shipped expression on an admitted 1..10 grid reports 0.01794 of
   the reference — a 98.2% silent underestimate, reproduced
   digit-for-digit. Never-relabel + exact z=0 ownership +
   conditional advertisement + certified k-completeness persisted
   with the manifest + float64/radicand validation; the R = 8
   USER RULING stays open. Spec: families-background-mps.md
   "UNIT 2 EXTENDED (45M-67)".
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
   EXTENDED (45M-69, seventeenth batch): _passed is status-only
   (run_board.py:866-868) — a stored PASS survives ANY executable
   change (resume skip :952, dependency accept :960; records
   carry status/detail/ts only), so a green BOARD.md can certify
   a tree that never executed the gate. Executable-surface digest
   stored with every verdict, PASS reusable only on digest
   equality, legacy/digestless = STALE, dependencies
   current-digest only, digest surface = the SAME expanded
   surface as this entry's dirty-watch fix, propose-first digest
   design, eight CPU legs + status-only mutation arm. Interim:
   --force-rerun stays the manual truth control. Spec:
   gates-and-board.md "UNIT 4 EXTENDED (45M-69)".
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
   EXTENDED (twelfth wave, folded in per the red team's sequencing):
   output identity — the shared family driver's run tag
   (`<model>[_t<T>]_ntrain<N>`) is not an artifact identity, so CMB
   TT/EE, BAOSN Hubble/D_M, MPS pklin/boost, and same-temperature cs
   probes all collide on the documented default path and save_emulator
   silently destroys the older valid pair; identity must come from
   resolved scientific facts + a config/dataset digest, with
   preexistence refusal; must land WITH the transactional publication
   clause (atomicity alone makes the wrong replacement atomic) and
   BEFORE the five-artifact production step. Spec:
   artifacts-inference-warmstart.md (+ its twelfth-wave extension).
   VERIFIED (Fable, 2026-07-12, twelfth wave): run_tag =
   cosmic_shear_train_emulator.py:193-217 (t<T> from
   `re.search(r"_cs_(\d+)")` on the train-dv FILENAME), save root
   :365 with --save default "emulator", diag same tag :464; thin
   drivers share main (cmb :44, baosn :45, mps :47) and their example
   commands pass no --save; scalar tag scalar_train_emulator.py:95-114;
   overwrite silent at results.py:232 (torch.save) + :259
   (h5py.File "w"); the "so runs do not overwrite each other"
   docstrings at cs :202 and scalar :15/:103 are false across
   products.
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
   Stubbed identity gates miss all three. EXTENDED (eleventh wave,
   folded in per the red team's sequencing): value-schema totality
   across all five adapters (quoted-"false" compile, the ROOTDIR ""
   default, untyped `emulators` — a string iterates per character —
   coerced dv_return/lmax), the multi-emulator composition defect
   (blind np.concatenate: two full vectors serve as one 2N vector;
   section mode never checks probe/section disjointness), and the
   CMB must_provide int() coercion. Spec:
   artifacts-inference-warmstart.md (the wave-2 section + its
   eleventh-wave extension).
   VERIFIED (Fable, 2026-07-12): emul_mps.py get_can_support_params
   returns ['Pk_grid', 'Pk_interpolator', 'sigma8'] (the input-param
   hook; products belong in get_can_provide / must_provide);
   emul_scalars.py calculate writes derived only under
   `if want_derived and "derived" in state` (real cobaya never
   pre-creates the key — the theory must); emul_mps.py hard-codes
   req["As"] while syren_params_from (syren_base.py) accepts
   As_1e9 | As, and the training dumps themselves sample As_1e9
   (analytics.py reads that column by name).
   AMENDED (45M-45, fifth batch): the alias-consistency boundary —
   dual As_1e9/As must satisfy As_1e9 == 1e9*As, dual w/w0 must agree
   numerically, else the shared reader raises naming both;
   canonicalize only after the proof; the adapter fails clean (no Pk
   state), the generator rejects pre-write. Reproduced live on the
   Mac (amplitude conflict 0.7667 max rel, dark-energy 0.2449, all
   spectra finite+positive — downstream guards blind). Spec:
   artifacts-inference-warmstart.md.
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
   EXTENDED (45M-68, seventeenth batch): the loader VERIFIES the
   .paramnames sidecar then discards the resolved names and
   slices [:, 2:-1] positionally (pool_size repeats the literal
   slice); reproduced through the real load_source — two derived
   columns give a 3-wide C against a 2-name covmat, zero derived
   silently drops a sampled parameter; _scalar_columns already
   resolves by name in the same file. One shared named-column
   resolver for load_source/load_scalar_source/pool_size/
   checkpoint-reload/generator-readback. Spec:
   data-generation-and-cuts.md "UNIT 8 EXTENDED (45M-68)".
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
   EXTENDED (thirteenth wave, folded in per the red team's
   sequencing — distinct from unit 14 because finite model outputs
   are the control condition): the diagnostics CREATE NaN from finite
   inputs and publish it — coverage_diagnostic medians empty classes
   (all-good -> median_bad NaN; the NaN verdict comparison then
   prints "not clearly coverage", a PERFECT run reported as
   ambiguous); hard_direction_regression 0/0 R^2 when every dchi2 is
   floored, corrcoef NaN on constants, the omega-baryon g.std()
   unguarded (the generic features ARE guarded — the asymmetry);
   cmb_residual_diagnostic divides by truth unmasked, a TE zero
   crossing NaNs all five fractional bands. No gate runs the real
   functions (the four smokes hand-build finite coverage dicts).
   Contract: explicit status/reason + counts, never numeric NaN
   sentinels; spectrum-aware validity mask; save-first order STAYS.
   VERIFIED (Fable, 2026-07-12, thirteenth wave): diagnostics.py
   :123-129 (empty-class medians + NaN-False verdict), driver
   :483/:487-488 (formats NaN, prints the negative finding), :293/
   :315 (0/0 R^2), :310 (corrcoef), :302-vs-:326 (guard asymmetry),
   :320-331 (NaN already a documented sentinel for r2_omega), :420/
   :424 (unmasked frac + all-NaN bands), :354-355 (TE crossing
   acknowledged, no mask), driver :358 (save-first deliberate),
   bsn_smoke.py:341 / cmb_smoke.py:484 / scalar_smoke.py:203 /
   mps_smoke.py:265 (hand-built coverage dicts); grep
   coverage_diagnostic|hard_direction over gates/ = zero hits.
   Spec: training-stack.md (the generic-diagnostics section + its
   thirteenth-wave extension).
   AMENDED (45M-51, seventh batch): the scalar driver re-declares
   eligibility from its family name — scalar NPCE is legal but
   scalar_train_emulator.py:267 calls local_linear_floor
   unconditionally behind ':251 the scalar loss is a plain chi2'
   prose while diagnostics.py:185 refuses needs_params losses, so
   every scalar NPCE --diagnostic run trains, SAVES, then raises
   instead of producing the PDF; the shared driver's :500
   capability branch is the correct rule. Capability-owned
   eligibility + structured availability record; spec appended in
   training-stack.md.
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
    AMENDED (45M-49, sixth batch): the float32-resolution guard is a
    float64 RELATIVE test while the constructor stores float32 —
    it accepts columns whose STORED scale is exactly zero
    (reproduced: targets [0, 0, 1.4e-45] pass the guard and store
    center 0.0 / scale 0.0; encode divides by zero). Stored-
    representation validation of every center/scale pair, the
    grid2d const-mask decision, and the covariance sqrt scales;
    spec appended in artifacts-inference-warmstart.md.
    AMENDED (45M-65, fifteenth batch): parameter-side covariance
    ingestion joins — from_covmat is loadtxt -> eigh with ZERO
    validation (a valid 1x1 covmat loads 0-dimensional and eigh
    raises before training; a NEGATIVE variance builds
    sqrt_ev = [nan, 1.0] silently — both proven through the real
    from_covmat); np.cov one-feature and
    AmplitudeFactorGeometry.from_covmat share the class. Spec:
    artifacts-inference-warmstart.md "UNIT 11 AMENDED (45M-65)".
12. **Optimized-mode validation parity.** Seventeen runtime guards across
    batching, model designs, geometry, and loss code are `assert` statements
    and disappear under `python -O`. Spec: conventions-and-workflow.md.
    VERIFIED (Fable, 2026-07-12) as a class; the census at this HEAD is
    EIGHTEEN `^assert` statements (batching 1, designs/ia 6,
    designs/plain 6, designs/blocks 1, losses/core 1,
    geometries/output 2, geometries/parameter 1), all config/geometry/
    user-data guards. The red team concurs (2026-07-12): seventeen was
    a counting error on their side; eighteen stands at 4f4dab3 and at
    the reviewed HEAD. (This paragraph was misfiled under entry 28 by
    an earlier edit; restored here 2026-07-12.) The ia.py/blocks.py
    entries of this census are satisfied by unit 29's typed-exception
    clause — cross-reference, no double work.
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
    AMENDED 45M-01 (2026-07-12; the red-team rounds are now labeled
    45M-XY by user convention, and Implementer handoffs carry the
    label): the fiducial params block has the same disease —
    validate_lcdm_params (compute_cmb_covariance.py:334-378) checks
    only PRESENT entries (no required keys, no exclusive-alternatives
    rule), bool passes the numeric check (subclass of int), and NaN
    defeats the omk pin (abs(nan) > 1e-12 is False). Probe-confirmed
    7/7 on the exec-extracted shipped body: empty mapping, As: true,
    As: .nan, omk: .nan, no amplitude at all, As+logA, and
    H0+thetastar are ALL accepted. The unresolved mapping goes to
    cobaya (:405) and to provenance (:751), so the covariance file
    cannot reconstruct the fiducial cosmology that generated its own
    spectra. Schema (exactly-one amplitude/expansion, required
    singletons, finite non-bool values, omk/w/wa explicit) +
    resolved-mapping persistence folded into this unit's spec:
    families-scalar-cmb.md, "45M-01 amendment".
14. **Finite training/evaluation contract** (red-team fourth wave,
    2026-07-12, CRITICAL — JUMPS THE QUEUE to right after the
    in-flight fixture unit: it protects every production training,
    and the five artifacts are next). NaN chi2 scores as a perfect
    emulator: frac counts NaN below every threshold, the best-epoch
    rule snapshots NaN weights, no finite guard on loss/grad/step,
    and the dead-network smoke bars PASS a NaN run. EXTENDED
    (eighth wave, folded in per the red team's sequencing): the
    pre-training PARITY verdict has the same hole — build_warm_start
    prints "[ok] ... max|dv| = nan" (NaN > tol is False;
    warmstart.py ~863), and the incidental torch.equal catch is
    skipped when n_extra == 0. Spec:
    training-stack.md, "NaN scores as a perfect emulator"
    (including the pre-training parity clause).
    VERIFIED (Fable, 2026-07-12): training.py ~1427-1433 (median
    propagates NaN, (c > t) counts NaN below), ~2047 (f0 = 0.0 wins),
    ~1971-1980 (backward/clip/step unguarded), zero isfinite/isnan
    in the file.
    REOPENED (45M-53 + addendum, eighth batch): the increment-(c)
    guard is training-only — eval_val (training.py:1490) and
    eval_source_chi2 (:1572) accept a finite NEGATIVE chi2, which
    counts as a PERFECT row in the threshold fractions and can
    crown the corrupted epoch; the gate's compiled arm greens on
    ANY compile exception (finite_contract.py:703-712) while the
    board advertises eager/compile agreement. Increment (e): one
    shared chi2-domain predicate at all three boundaries + eval
    raise + ADJUDICATED scale-aware band max(1e-6, 32*eps*n_terms)
    superseding the absolute constant + compiled-leg capability
    truth (CUDA lane red-on-exception, non-green skip only).
    Fifth-sqrt-site flag CONFIRMED. (e) runs before 42+43.
    Spec: training-stack.md, "UNIT 14 REOPENED (45M-53)".
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
    AMENDED (45M-54, ninth batch): the mps-smoke range leg
    (mps_smoke.py:394-398) reports PASS on ANY exception — zero
    leg-level catch power for the refusal contract the vendored
    check_ranges actually provides (LoggedError naming coordinate,
    requested value, stored limit). Amendment: in-range control
    first, all four boundaries, LoggedError-only with the message
    pinned, wrong-class mutation arm. Spec:
    families-background-mps.md "UNIT 16 AMENDED (45M-54)".
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
18. **Schedule validation + direction-correct step** (fifth wave).
    trim/focus schedules reach anneal_value unvalidated (only the
    berhu/ema anneal sub-blocks pass _validate_anneal_block); an
    unknown shape silently runs LINEAR; anneal_epochs 0 is silently
    max(1,...)'d; the step arm is decreasing-only — an increasing
    ramp jumps to end at the first ramp epoch. Spec:
    training-stack.md, "Schedule validation".
    VERIFIED (Fable, 2026-07-12): losses/core.py anneal_value
    ~30-81 (fall-through to linear; max(end, floor) picks end on an
    increasing ramp); no trim/focus range validation exists.
19. **NPCE absolute LOO gate** (fifth wave, CRITICAL — the full
    contract for wave-1 unit 5b's pce fallback). `if not cols:
    always keep mode 0` persists a failed mode while the report
    claims the predicate held; select_lars_loo's all-active score
    vector is -1 everywhere so argmax duplicates column 0. Spec:
    models-and-designs.md, "NPCE LOO gate must be absolute". Land
    BEFORE any NPCE production training.
    VERIFIED (Fable, 2026-07-12): pce.py ~414-419 and ~209-211,
    both by direct read.
20. **Hyperparameter-range validation** (fifth wave). The
    [default, min, max, kind] machinery validates nothing:
    out-of-bounds defaults train silently, int() truncates,
    reversed bounds reach suggest_int, log accepts a zero lower
    bound, a kind typo demotes the range to a fixed value. Spec:
    training-stack.md, "Hyperparameter-range validation". Land
    before production tuning sweeps.
    VERIFIED (Fable, 2026-07-12): training.py _range_default
    ~674-677 (int truncation, no bounds), _suggest_range ~680-692
    (unvalidated lo/hi to Optuna).
21. **Inference numerical boundary** (fifth wave). _as_row checks
    names/counts only (NaN/Inf/bool enter the model; decoded
    NaN/Inf served); the CMB amplitude law accepts As <= 0 and
    NaN tau. Spec: artifacts-inference-warmstart.md, "Inference
    numerical boundary". Land BEFORE the EMUL2 acceptance.
    VERIFIED (Fable, 2026-07-12): inference.py ~442-457 (documented
    raises are KeyError/length only), losses/cmb.py ~316 (no domain
    check in _factor).
22. **Selection-record truth** (fifth wave, CRITICAL — the full
    contract for wave-1 unit 3). The loop restores the epoch-0
    baseline but histories carry trained epochs only, so drivers /
    artifact attrs / Optuna report a DIFFERENT model than the one
    shipped; same at trunk->head and refine boundaries. Spec:
    training-stack.md, "Selection-record truth". Pairs with unit 14
    (its metrics ride the finite contract).
    VERIFIED (Fable, 2026-07-12): histories start empty ~1694, the
    baseline eval ~1844 seeds best_* without appending; the driver
    argmin recompute was verified in wave 1.
23. **Run-control schema totality** (sixth wave). No top-level
    train_args whitelist (a `clipp` typo silently changes training
    via the consumers' .get()/signature defaults); the phase
    whitelist validates names, not values ({"clip": NaN},
    {"rewind": "false"} pass); `if clip > 0.0` silently disables
    clipping on NaN/negatives; a quoted "false" enables rewind;
    bs/nepochs totality still unguarded. EXTENDED (seventh wave,
    folded in per the red team's sequencing): the config ROOT is
    also unwhitelisted — pcee:/transferr: typos are inert extra
    data and the run silently trains a plain emulator (all feature
    reads are cfg.get(): experiment.py ~589/721/736/1239). Spec:
    training-stack.md, "Run-control schema totality" (including
    the root-level clause) — bundles with units 18 + 20 (+ 29,
    the model-block value schema, fourteenth wave) as one
    train_args-totality cluster.
    VERIFIED (Fable, 2026-07-12): validate_phase_block ~492-522
    (eight-key whitelist, structural checks only), loop ~1977
    (clip > 0.0), ~1866 (if rewind:), signature defaults
    ~1513-1514; root reads all cfg.get(), no root whitelist
    anywhere in experiment.py.
24. **Fine-tune anchor truth** (eighth wave, a training-truth unit —
    with or immediately after the 14+22 pair, before any anchored
    production fine-tune). The documented finetune.anchor is
    config-BLOCKED by a stale "not implemented in V1"
    NotImplementedError while the downstream facility exists and a
    `>= 0` validator sits unreachable behind it; once unblocked, the
    compiled-CUDA `_orig_mod.` name prefix makes build_anchor match
    ZERO parameters silently (the artifact then records an anchor
    that never ran). The README/example advertised the key as
    available — CORRECTED by the Architect the same day (docs now
    say "currently refused"); the unit restores the published
    contract. Spec: artifacts-inference-warmstart.md, "Fine-tune
    anchor truth".
    VERIFIED (Fable, 2026-07-12): warmstart.py ~159-163 (the
    unconditional raise), ~182-186 (unreachable validator), ~58
    (whitelisted key), ~86/271 (source rebuilt eager);
    training.py ~306-311 (the silent skip build_anchor inherits);
    README ~1683 advertised anchor: 1.0e-2 (now corrected).
    AMENDED (45M-50, seventh batch): scalar fine-tuning executes,
    then erases its source provenance at save — validate_scalar
    permits it, the scalar from_config branch sets
    _finetune/_finetune_root, but scalar_train_emulator.py's own
    attrs (:201-211) never inspect exp._finetune while the shared
    driver writes finetuned_from/finetune_extra_names (:385-387;
    untruncated grep: the shared driver is the ONLY writer). One
    shared artifact-provenance assembler for every driver;
    executed-anchor record shares the path once anchors open;
    spec appended in artifacts-inference-warmstart.md.
25. **Nested data paths never resolve** (ninth wave).
    resolve_cocoa_config rewrites only the flat data keys; the
    nested cmb.covariance / grid.z_file / grid2d file leaves stay
    cwd-relative, splitting every shipped CMB/background/MPS
    example across two path bases; the docstring's "every
    data-block file path" is false. Spec:
    data-generation-and-cuts.md, "Nested data paths never resolve".
    VERIFIED (Fable, 2026-07-12): cocoa.py ~135-139 (the flat-keys
    loop), docstring ~81-82.
26. **Validation grid axes are never identified** (ninth wave).
    One shared z_file/k_file interprets both train and val dumps
    ("val borrows the" training axes, experiment.py ~3145); a
    same-width val dump or base from another run scores silently
    on the wrong grid. Spec: data-generation-and-cuts.md,
    "Validation grid axes are never identified". CLUSTER RULING
    (red-team, adopted): 8 + 17 + 25 + 26 = one file-set
    authenticity boundary, taken together.
    VERIFIED (Fable, 2026-07-12): single-axis config schema, the
    staging comment, width-only checks.
27. **Bounded grid2d staging: audit outcome.** The Implementer's
    unit (8 files, uncommitted at cb4f1f1) is STRUCTURE ACCEPTED,
    REVISION REQUIRED before landing: the streamed variance is the
    naive (s2 - s1*s1/n)/n with a zero-clamp — order-dependent
    (Architect probe 3.9659 vs red-team probe 4.1279 vs true 4.0 on
    the 1-ULP fixture) and able to false-pin varying columns; the
    revision replaces it with per-chunk mean/M2 Chan/Welford in
    float64 + red legs that FAIL the s1/s2 form. AMENDED at the
    codex merge with the red team's float32-payload clause (binding):
    the accumulator is fed the exact float32 rows written to the
    result, promoted to float64 — never the pre-cast float64 chunk —
    and the gate reference is the former materialized-float32 path.
    The from_stats scope deviation is CONFIRMED in scope; the
    folded-in transfer-fixture fix is ACCEPTED verbatim; the
    human-facing "oracle" prose rider joins the revision. Full
    verdict + adjudicated amendment:
    data-generation-and-cuts.md, "Bounded-staging Architect audit".
    RE-AUDITED (Fable, 2026-07-12): the revision's NUMERICS are
    ACCEPTED — independent probe 23/24 (Chan merge == np.std(ddof 0)
    across chunkings at rtol 1e-9; the old form reproduces the false
    pin, column-0 std exactly 0.0 at chunk height 7; the
    float32-payload clause discriminated at a 12% stored-vs-precast
    gap). TWO AMENDMENTS before the commit: (1) check_stable_moments
    leg 2 CRASHES the gate — its 1-ULP column sits below the shipped
    RELATIVE pin threshold (8*eps32*|center| ~ 95 at 1e8), both
    columns classify constant, and the whole-surface dead-dump
    ValueError fires (reproduced with the leg's exact numbers); the
    fixture needs an above-threshold varying column (e.g. +-1024).
    (2) The _grid2d_law_rows shape-flow diagram + legend still name
    the deleted s1/s2 accumulator. Full verdict:
    data-generation-and-cuts.md, "Revision re-audit".
    AMENDMENTS RE-CHECKED + UNIT COMMITTED (Fable, 2026-07-12): leg 2
    is the three-column fixture ([True, True, False] verified through
    the shipped from_stats, no dead-dump crash), the diagram names
    (count, mean, M2), the other five files byte-identical to the
    accepted delivery. Committed c03a084.
    CLOSE REOPENED (red team, Architect-ADJUDICATED, same day): the
    disk-backed staging files LEAK across sweep points — mkstemp +
    atexit-only, no ownership; the shared N-train sweep reuses one
    experiment per lane and orphans each point's ~4.57 GiB .g2law.dat
    until worker exit; the gate's disk leg checks only
    isinstance(memmap). My deviation-2 acceptance was scoped to a
    single run and is CORRECTED. Numerics stand. Micro-revision
    (slots on the experiment, supersede-on-restage, public
    release_train/val_staging for the sweep lane, failure-path
    unlink, five red legs) must land BEFORE the three-gate rerun is
    spent. Spec: data-generation-and-cuts.md, "Close REOPENED".
    MICRO-REVISION LANDED (2026-07-12, Implementer;
    Architect-audited ACCEPTED): ownership slots + returned temp
    path + supersede-on-restage + public release_train/val_staging +
    sweep-lane release (:174, after the try/except) + failure-path
    unlink + check_staging_lifecycle (five legs, mps-identity).
    Independent probe green over the exec-extracted shipped bodies;
    AST fan-out proves the gate fake cannot AttributeError on the
    workstation. One rider (not a hold): the board.py mps-identity
    `maps` string still omits the lifecycle leg — fold into the next
    Implementer commit touching gates/. The amended close is now
    SPENDABLE: the three-gate rerun (mps-identity, cmb-identity,
    transfer-identity; 32/32) is user-run on the workstation. Audit
    record: data-generation-and-cuts.md, "Audit (2026-07-12, Fable):
    ACCEPTED".
28. **Validation leakage + data-control totality** (tenth wave,
    CRITICAL first clause, folded into the 8+17+25+26 file-set
    authenticity cluster per the red team's recommendation).
    stage_train/stage_val share one split_seed with NO train/val
    alias or disjointness check — the stage_val docstring states the
    unenforced premise verbatim; aliased paths make validation BE
    training. Plus: one-row files crash in parsing (raw loadtxt,
    no atleast_2d) though validate_sizes permits n = 1; split_seed /
    ram_frac / param_cuts bounds are coerced or key-checked only.
    Spec: data-generation-and-cuts.md, "Tenth wave: validation
    leakage + data-control totality".
    VERIFIED (Fable, 2026-07-12): experiment.py ~3086/~3142/~3169
    (same seed), ~3126-3127 (the docstring premise), no
    samefile/duplicate check anywhere; data_staging.py ~536/~779
    (raw loadtxt 2-D indexing vs generator_core ~610's atleast_2d);
    bare int()/float() control reads.
29. **Model-block value schema** (fourteenth wave; joins the
    train_args-totality cluster 18+20+23, distinct from
    range-SYNTAX validation — fixed YAML values have the same
    holes). The nested model: schema validates key names only;
    values flow untyped into the design constructors. Headline:
    `model.trf.n_blocks: 0` is a SILENT ARCHITECTURE DEMOTION — the
    identity-start doctrine makes corr = t - t0 identically zero, the
    trunk trains, collapse bars can pass, and the requested
    transformer head never exists. Plus: quoted-"false" truthiness
    (rescale_kernel/separable/film/shared_mlp), zero-block
    IndexErrors with unrelated messages, n_heads 0 dividing by zero
    inside its own assert (gone under -O), float(gate_init)
    accepting NaN/Inf (Inf * corr-0 = NaN at step one), int()
    coercions on n_gates (both activation paths) and n_tokens.
    Contract: one pure active-model value validator before
    construction; inactive-blocks-stay ruling preserved; constructor
    asserts on public config paths become typed exceptions
    (SATISFIES unit 12's ia.py/blocks.py census entries —
    cross-reference, no double work). Spec: models-and-designs.md,
    "Model-block value schema".
    VERIFIED (Fable, 2026-07-12, fourteenth wave): whitelist
    key-names-only at experiment.py:225; ResTRF forward t == t0 at
    ia.py:933-947; convs[-1] at ia.py:505; mlp_lins[-1] at
    blocks.py:639; `assert dim % n_heads == 0` at blocks.py:602;
    kernel/groups/geometry asserts at ia.py:353/:416/:423/:448;
    float(gate_init) at ia.py:490; int(n_gates) at
    experiment.py:292/:4108/:4258/:4267; int(n_tokens) at
    plain.py:866.
30. **plot_xi plotting truth** (45M-02, 2026-07-12). The colorbar is
    built from Normalize(param[0], param[-1]) but every curve is
    colored cm(index / n) — the parameter value is never used; marker
    edges repeat the wrong mapping; the style cyclers advance
    globally across panels; malformed input prints "Bad Input" and
    returns int 0; no gate calls plot_xi (full grep: zero callers
    outside plotting.py). Folds in the hardening-ledger mutable-list
    defaults on the same signature (no second unit). Contract + the
    Agg-backend board leg (permutation arm, multi-panel arm, RGBA
    inspected against the colorbar's ScalarMappable): spec in
    training-stack.md, "plot_xi does not draw the colors its colorbar
    describes".
    VERIFIED (Fable, 2026-07-12): plotting.py:1647-1648 (norm),
    :1755/:1761/:1770/:1776 (index colors + marker edges),
    :1667-1677 (global cyclers), :1620/:1623/:1663 ("Bad Input" -> 0;
    the length check runs after the colorbar is drawn), mutable
    defaults on the signature (:1566-1569).
31. **Generator entry files self-teaching** (45M-03, 2026-07-12;
    documentation repair, no new note file). All four
    dataset_generator_* files have no module docstring
    (AST-verified) and _compute_dvs_from_sample — the family physics
    boundary — has no formal docstring in any of them; store hooks
    uneven. Module docstrings with the flow diagram, the subclass
    contract stated locally, formal blocks on every override hook,
    banner walls replaced. SEQUENCE AFTER unit 33 (the component
    loop it documents is being rewritten). Spec:
    conventions-and-workflow.md, "Generator entry files must be
    self-teaching".
32. **Eval-batch gate stops executing at import** (45M-04,
    2026-07-12). gates/checks/ge_c_eval_bs.py allocates C/DV/model/
    loss/thresholds at module scope, closes helpers over them, runs
    the whole test at import, and sys.exits (:149) — the one
    production exception to the no-global-data rule; the board
    survives only because board.py:369 runs it as a subprocess.
    main-ification with explicit-parameter helpers, numbers
    preserved, board still executing the real check. Spec:
    gates-and-board.md, "The eval-batch check must stop executing at
    import".
    VERIFIED (Fable, 2026-07-12): module scope :33-:60, sys.exit
    :149, "there is no main" docstring :4, board.py:369 run_check.
33. **Generator physics-execution truth** (45M-06, 2026-07-12;
    joins the file-set/ingress campaign — cluster now
    8+17+25+26+28+33). lensing:103, cmb:307/:327, mps:364 zip
    component execution against the unused private
    _params_of_dependencies (silent truncation on any length
    mismatch, cached=True); the same hand-built lifecycle already
    produced the bitwise-constant H(z) dump (background switched to
    logposterior(cached=False) at :335 — the worked reference);
    lensing:99 picks the truth likelihood by YAML insertion order.
    Contract (public lifecycle preferred, identity-selected
    producer, provenance-recorded, fake-Cobaya pure legs + mutation
    arm + two-cosmology workstation smoke): spec in
    data-generation-and-cuts.md, "Generator physics execution".
    NB: the existing "TWO evaluation idioms, deliberately
    unharmonized" paragraph in that note is SUPERSEDED by this
    contract's preferred path — harmonization now has its gate
    evidence plan; the paragraph gets rewritten when 33 lands.
    AMENDED (45M-64, fifteenth batch): the worked reference is
    itself verdict-blind — the returned LogPosterior is DISCARDED
    at background:335 and success = absence of three keywords in
    captured terminal text; a rejected point (logpost = -inf
    returned without raising, proven on real cobaya 3.6.2) reads
    the provider unconditionally and can serve the previous
    point's finite physics as a success. Acceptance-helper
    contract + fake-Cobaya order/token legs; background is the
    FIRST PATIENT, not the template. Spec:
    data-generation-and-cuts.md "UNIT 33 AMENDED (45M-64)".
    AMENDED AGAIN (45M-70, seventeenth batch): the SAME
    verdict-blind pattern at all six gate-side lifecycle sites
    (cmb_smoke :441, bsn_smoke :248/:291, mps_smoke :331/:376)
    and the CMB covariance producer (:420) — the gates prove "a
    value was readable", not that the point was accepted; ONE
    shared acceptance definition (first landing establishes it,
    gate legs ride the wave-4 family gate visits, the covariance
    producer rides the 33 helper), rejected point = zero getters
    = red, CAMB-reference rejection never reference truth. Spec:
    data-generation-and-cuts.md "UNIT 33 AMENDED AGAIN
    (45M-70)".
34. **Public prose states the current state** (45M-07, 2026-07-12;
    documentation-only). emulator/README.md:216 narrates board-run
    history; warmstart.py:162 names "unit 2" in a user-facing error;
    all-caps emphasis persists (README INPUT :24, ONE
    :68/:136/:183/:347). History moves to gates-and-board.md; the
    warmstart error becomes actionable current-state prose;
    emphasis capitals become definitional wording; acceptance = the
    prose diff + a complete untruncated scan. Spec:
    conventions-and-workflow.md, "Public prose states the current
    state". Batches naturally with unit 31 (one documentation
    handoff), AFTER 33. EXTENDED with the 45M-17 fold-in (the
    CosmolikeChi2.decode docstring claims a scattered
    (B, total_size) return but the geometry decode returns the KEPT
    width; verified losses/core.py:165 vs geometries/output.py:506;
    docs-only + subclass/caller audit; spec fold-in in
    conventions-and-workflow.md).
35. **Memory probe truth** (45M-11, 2026-07-12). The sizing forward
    mutates model state (BatchNorm buffers, RNG) in the model's
    current mode with a DEFAULT-dtype zeros batch;
    compute_model_size_bytes multiplies total numel by the FIRST
    param's element_size, counts no buffers, sees no loss-owned
    resident state — and the number picks the placement regime.
    Contract + torch gate: training-stack.md, "The memory probe is
    not observational". VERIFIED: batching.py:90 (zeros, default
    dtype), :121 (live forward, no state save), :140-169 (first-dtype
    multiply, params only).
36. **BAOSN odd-node quadrature REOPEN** (45M-12, 2026-07-12,
    CRITICAL — the red team's top priority; FIRST code unit after
    the finite contract completes). The odd-node "cumulative
    Simpson" increment dz/6*(1,4,1) (background.py:85) is HALF the
    two-interval Simpson total, not the one-interval integral —
    Architect-reproduced: y=z error EXACTLY h^2/2 at every h
    (first order; the recorded O(dz^3) claim is FALSE); y=z^2 gives
    4h^3/3 vs h^3/3; the (5,8,-1)/12 replacement is exact on both;
    interpolators fit through the wrong odd nodes so arbitrary-z
    queries are contaminated; bsn_identity.py:87-88 encodes the bug
    as acceptance (e_odd < 1e-3). The prior bug-for-bug porting
    acceptance is REOPENED (it rested on the false error order).
    USER-VISIBLE: served BAOSN values change between grid nodes;
    the served-value comparison reruns. Spec:
    families-background-mps.md, "REOPENED: the odd-node cumulative
    Simpson".
37. **Implementation-identity manifest** (45M-13, 2026-07-12; joins
    the artifact-integrity campaign beside unit 3 — implementation
    identity distinct from pair identity). git_commit is written
    (results.py:398) and documented as validated (:192 "rebuild
    refuses one without") but rebuild never reads it; the recipe
    imports CURRENT code under stored names; the syren-law MPS
    artifact serves P * (base_new / base_old) if the vendored base
    changes. Compatibility manifest + versioned registry + refuse-
    or-migrate on rebuild + the MPS/syren base binding + honest
    git_commit docs. Spec: artifacts-inference-warmstart.md,
    "Artifacts are not bound to the code".
38. **Syren fit-domain validation** (45M-14, 2026-07-12).
    log(abs+eps) (linear.py:31/:186/:226/:232/:373-375) and
    sqrt(abs) (syrenhalofit.py:317) reflect out-of-domain physics
    into valid-looking numbers; Architect-reproduced LIVE: the
    -0.04754964 radicand at sigma8 1.2/Om 0.1/h 0.5/ns 0.8/a 0.1
    returns 0.11591803 on the vectorized path while the scalar
    sibling NaNs — the two disagree at the boundary. Domain
    documented + validated before evaluation, abs/eps continuation
    removed from production, scalar/vector agreement, prior+grid
    pre-launch validation, base variant recorded in unit 37's
    manifest. Spec: families-background-mps.md, "Syren silently
    reflects out-of-domain physics".
39. **make_chi2 total dispatch** (45M-15, 2026-07-12). "residual"
    works only as the catch-all else (losses/core.py:839-847) —
    every typo/None/object selects ResidualBaseChi2, and
    build_shear_angle_map (filesystem) runs BEFORE mode validation.
    Three equality branches + ValueError + arg validation before
    the angle-map + real-bool include_amp + byte-identical valid
    modes. Spec: training-stack.md, "make_chi2 turns every unknown
    rescale mode into the residual algorithm".
40. **Power-activation zero Jacobian** (45M-16, 2026-07-12; red-team
    priority 3). psi = sign(x)*((1+|x|)^p - 1)/p
    (activations.py:147/:220) has autograd derivative 0 at exactly
    x=0 (documented: 1; full default activation: 0 vs H's 0.5) — a
    gradient-absorbing point exactly where the identity-start
    doctrine places zeros; invisible to forward checks. Reimplement
    as x * even-ratio with analytic limit 1, stable near-zero,
    p-schema, derivative legs + float64 gradcheck + zero-Jacobian
    mutation assert. Spec: models-and-designs.md, "The power
    activations have a zero derivative at exactly zero".
41. **Resolved-pass record** (45M-19, 2026-07-12). resolved_train
    stores RAW inputs on the flagship never-trust-defaults surface
    (training.py:3028 "loss": loss — null when the run consumed the
    default sqrt mode; per-phase effective passes and refinement
    inheritance unrecorded; nepochs pre-refinement while histories
    include it). Persist the passes sequence in execution order +
    refinement pass + total_epochs == history rows + raw YAML kept
    separately + readers read the record. Spec:
    artifacts-inference-warmstart.md, "config_resolved_yaml does not
    record what the run consumed".
42. **CMB amplitude-law metric REOPEN** (45M-21, 2026-07-12,
    CRITICAL). CmbFactoredChi2.chi2 (losses/cmb.py:371-380) ignores
    params_whitened and delegates to the diagonal sum-of-squares;
    the docstring's "the factor cancels in the residual" is FALSE —
    the residual is f*(C_pred - C_truth)/sigma, so the reported chi2
    is f^2 * chi2_physical with f = exp(2 tau)/A_s (:334): selection
    biased toward small-f cosmologies, the 0.2 threshold detached
    from the covariance metric, roughness's law-neutral claim false
    too. The identity gate only round-trips; it never checks a
    physical known answer. Fix: divide the whitened residual by f in
    chi2; rerun cmb-identity + cmb-smoke. Spec:
    families-scalar-cmb.md, "REOPENED: the CMB amplitude law",
    queue-42 half.
43. **CMB amplitude-factor normalization REOPEN** (45M-22,
    2026-07-12, CRITICAL, coupled to 42). Raw Cobaya A_s ~ 2.1e-9
    makes f ~ 5e8 at fiducial — a 1e9-scale network target with a
    float32 center subtracting nearly equal values (a unit-porting
    defect: legacy 1e9*A_s replaced by raw A_s without the reference
    normalization). Fix: dimensionless f = (A_s_ref/A_s) *
    exp(2(tau - tau_ref)), f = 1 at the persisted fiducial;
    as_ref/tau_ref resolved artifact facts; a NEW semantic law
    version (unit 37's manifest distinguishes; old artifacts
    REFUSED; affected CMB artifacts retrain). Spec: same section,
    queue-43 half.
44. **stream_stats Chan unification** (45M-23, 2026-07-12; amends
    the stable-moments standard — one numerical-statistics design
    repo-wide). data_staging.py:113-129 still ships the prohibited
    s1/s2 variance; Architect-REPRODUCED: offset 1e8 gives std
    1.8103 vs true 1.0017 silently, offset 1e10 gives NaN. Latent in
    the ordinary path (load_source keeps only dv_mean — the red
    team's own scope note), but a public normalization function
    returns false/NaN scales. Chan/Welford + full input schema +
    catch-power leg. Spec: data-generation-and-cuts.md, "The old
    unstable variance survived".
45. **Scheduler execution protocol** (45M-25, 2026-07-12; joins the
    run-control/train_args-totality campaign, DISTINCT from the
    schedule-VALUE unit). make_scheduler advertises any class
    (training.py:438-441) but the loop implements exactly two
    per-epoch protocols (:2082 step(median) for ReduceLROnPlateau,
    :2124 bare step() for everything else) — a OneCycleLR spec
    constructs and then runs the wrong schedule by orders of
    magnitude; per-phase/refine construction adds the step-horizon
    problem. Contract: restrict-and-refuse before model/loader
    setup, OR make cadence + argument protocol resolved scheduler
    facts stepped at the right place with true per-pass horizons
    (actual optimizer updates, chunk-tail rule included) and
    unambiguous warmup ownership; either way the resolved artifact
    records class, cadence, metric source, effective step count.
    Torch gate: counting schedulers (per-update / per-epoch), the
    plateau median argument, a tiny OneCycleLR known-LR sequence or
    startup refusal, no double warmup.
    UNIT 14 AMENDED (45M-47, fifth batch, increment d): finite batch
    losses publish an Inf epoch loss — loss.detach()*bs multiplies in
    float32 BEFORE the accumulator (training.py:2103; acc_dtype is
    float32 on MPS, :1781), so a finite 1e38 loss with bs=8 goes Inf
    and is appended/printed/persisted unguarded. RULING: host
    python-float (float64) accumulation reusing the per-step sync the
    finite contract already pays, + a required finite check on the
    completed epoch train_loss; extends the finite-contract gate
    (mutation arm restores the old ordering, must go Inf). Reproduced:
    np.float32(1e38)*8 -> Inf. Spec: training-stack.md. Unit 14 now
    closes on a+b+c+d + the extended gate.
    UNIT 14 AMENDED (45M-24, CRITICAL producer clause): the DEFAULT
    "sqrt" loss (losses/core.py:349; berhu lower branches :363/:381;
    the anneal's sqrt arm) has d sqrt(sum r^2)/dr = 0/0 -> NaN at an
    exact fit — identity-start heads, pinned grid2d columns, tiny
    fixtures, and zero corrections deliberately create exact zeros,
    and one such sample poisons the batch gradient. The finite guard
    (increment a) DETECTS this; the objective must stop PRODUCING
    it: one shared safe-sqrt transform (forward 0 AND gradient
    exactly 0 at c == 0, matching sqrt(c) for normal positives —
    sqrt(c+eps) is NOT contract-equivalent), used at all four sqrt
    sites; materially negative / nonfinite chi2 rejected before the
    transform (a scale-aware roundoff tolerance stated and tested if
    tolerated); C1 knot matching preserved. Unit 14 is now THREE
    increments: (a) training.py guards [landed, checkpoint],
    (b) warmstart parity, (c) safe-sqrt producer fix — and the
    dedicated finite-contract gate gains the 45M-24 red legs
    (exact-zero row per mode; mixed batch; pinned-column fixture;
    finite-and-zero gradients on the exact-fit row; analytic
    agreement on positives; negative/NaN chi2 refusal; eager +
    compiled).
    UNIT 15 AMENDED (45M-26): must_provide accepts
    Hubble={"z": [1090]} through the uniform union-window check
    (emul_baosn.py:214-236) while get_Hubble serves the SN grid only
    — accepted at startup, refused at runtime, invisible to the
    gate's uncrossed arms. One product-specific domain helper shared
    by must_provide and every getter; startup and runtime verdicts
    identical. Spec amendment: families-background-mps.md, "45M-26
    amendment to unit 15".
46. **NPCE domain policy** (45M-28, 2026-07-12; joins the
    inference-boundary campaign, unit 21, with explicit NPCE legs).
    PCEEmulator.forward clamps unconditionally to the Legendre box
    (pce.py:505-506) — one rounding unit outside and an arbitrarily
    distant cosmology collapse to the same boundary coordinate,
    finite and plausible, invisible to the finite guard; the refiner
    was trained around a base whose hidden saturation is part of its
    target. Named persisted domain policy (default: refusal with a
    documented tolerance; clamp is never the validator), identical
    policy at training/validation/inference, lo/hi schema,
    boundary-hit counts in the resolved record. DISTINCT from the
    LOO-selection unit (accepted as argued). Spec:
    models-and-designs.md, "NPCE maps arbitrarily out-of-domain
    cosmologies to the same boundary".
    UNIT 24 AMENDED (45M-29, BINDING before the anchor door
    reopens): the post-step order on HEAD is optimizer -> EMA
    (:1988) -> anchor (:1995) while the comment claims the opposite
    — the shipped/selected EMA samples the trajectory BEFORE each
    anchor pull (at beta = 0 it is the fully unanchored result).
    Canonical order optimizer -> anchor -> EMA; anchor-absent
    byte-identical; six red legs incl. the beta = 0 analytic leg and
    selection/readback replay. Spec: training-stack.md, "45M-29
    amendment to unit 24".
    UNIT 26 AMENDED (45M-27): CMB lmax validation proves only a
    stored MAXIMUM (emul_cmb.py:195-209) while calculate scatters
    into a zero array at stored ells (:247-249) — a gapped or
    late-start ell axis serves zeros indistinguishable from
    predictions. The axis-identity contract extends to the CMB
    read/rebuild boundary: ell must equal np.arange(2, ell[-1]+1)
    exactly, widths must match, the adapter validates before
    _ell_arrays, only l = 0,1 assembly-zero-filled; mutation leg on
    a same-shaped h5. Spec: families-scalar-cmb.md, "45M-27
    amendment".
47. **CMB dump multipole identity** (45M-30, 2026-07-12, CRITICAL;
    joins the file-set-authenticity cluster — now
    8+17+25+26+28+33+47 — and sharpens the 45M-27 amendment). The
    CMB generator writes four ANONYMOUS spectrum stores with no axis
    sidecar; training checks only dv.shape[1] == ell.size against
    the covariance (experiment.py:3549) and labels dump column 0
    with the covariance's first multipole — a same-width shifted
    lrange (10..1008 vs 2..1000) trains silently against the wrong
    covariance; the checkpoint loader compares no axis fact at all,
    so resume can reuse a stale same-width dump. Contract: a
    required ell sidecar in the generator file set; exact equality
    lrange == sidecar == every spectrum width == covariance ell at
    fresh/resume/append/training; np.arange(lmin, lmax+1) exact;
    sidecar in the transactional manifest; the artifact records the
    axis as dump-verified; never infer coordinates from filename or
    width. Spec: data-generation-and-cuts.md, "A CMB dump has no
    multipole identity".
48. **Finite-real validator predicate** (45M-31 + 45M-32,
    2026-07-12; joins the train_args-totality cluster — now
    18+20+23+29+45+48). Comparison-only validators pass NaN at five
    confirmed sites (EMA horizon ~1271, berhu knot/cap ~968,
    roughness lam/period ~1119, transfer.refine base_lr_scale
    ~1392 + anchor) — a NaN horizon poisons theta_bar on the first
    lerp, a NaN anchor corrupts every anchored base tensor on step
    one. ONE shared predicate (real, non-bool, finite, then domain)
    + a mechanical census of all comparison-only numeric leaves +
    trunk/head phase parity; runtime unit-14 guard stays defense in
    depth; the anchor integration legs fold into unit 24's gate.
    Spec: training-stack.md, "Comparison-only validators accept
    NaN".
    UNIT 41 AMENDED (45M-33): the transfer-refine drift metric
    iterates state_dict() so non-trainable buffers dilute the
    denominator, and a moved zero-reference tensor records relative
    drift 0.0 (undefined, not zero). Metric over trainable
    parameters via a canonical key set; norms + status persisted;
    key equality verified; summary recomputable; honest naming.
    Spec: artifacts-inference-warmstart.md, "45M-33 amendment to
    unit 41".
    UNIT 18 AMENDED (45M-34): the shared anneal whitelist permits
    shape const for every owner, but BerHu/EMA anneal force
    start=0,end=1 internally — const freezes s = 0 forever: berhu
    silently runs plain sqrt every epoch, and theta_bar is never
    allocated (the s > 0 activation guard) while the resolved record
    says EMA is on. const stays legal for trim/focus (start IS the
    constant); rejected for berhu/ema with an explanatory error;
    owner-parameterized shapes in the shared validator; catch-power
    leg proves the old const config demotes. Spec:
    training-stack.md, "45M-34 amendment to unit 18".
    UNIT 29 AMENDED (45M-35, BINDING, CRITICAL): "finite non-bool"
    still admits gate_init 0 — an exact absorbing state (correction
    is zero-init by design, so gate grad and every head-weight grad
    are both zero forever; the head never trains while collapse bars
    pass; ia.py:336-339 names the invariant unenforced). Rule:
    representably NONZERO after parameter-dtype conversion
    (float32-underflow rejected; positive-only deliberately NOT
    imposed); one validator over plain + factored heads; the
    demotion gate gains a behavioral one-step trainability leg
    (trunk frozen) — presence and trainability are separate
    requirements. Spec: models-and-designs.md, "45M-35 amendment to
    unit 29".
    UNIT 29 AMENDED AGAIN (45M-36, CRITICAL): the identity-at-init
    license ("every activation maps 0 -> 0") was incomplete — waking
    a zero-initialized final layer also requires representably
    nonzero a'(0). ReLU (derivative 0 at 0, torch convention) makes
    ResCNN/TemplateResCNN heads exact permanent dead branches and
    TRF blocks a PARTIAL demotion (attention wakes through the
    zeroed wo, the MLP half never does); the code comments claiming
    real first-step gradients are false for ReLU. Rule: a(0) = 0
    AND finite representably-nonzero a'(0) for zero-init-head
    activations; ReLU rejected in heads, legal in trunks (Architect
    ruling; no head-safe variant in V1); power/gated_power fold into
    unit 40's repair — pre-repair they fail the same check; the
    four misleading explanations corrected (the note's own claim
    already superseded in place). Spec: models-and-designs.md,
    "45M-36 amendment to unit 29".
49. **Optimizer execution protocol** (45M-37, 2026-07-12; joins the
    run-control campaign beside unit 45). make_optimizer and
    make_refine_optimizer advertise {cls, **kwargs} generality but
    unconditionally inject fused=True on CUDA for every class —
    overwriting an explicit fused: false (the spec is not forwarded
    as documented) — and the closure-free step() protocol is never
    validated (LBFGS constructs, then dies at step one). Ruling: the
    BOUNDED surface — closure-free Adam/AdamW-family; fused injected
    only where supported, user value preserved; closure-required
    rejected before construction; one shared capability decision for
    both factories; resolved class + fused state persisted (rides
    unit 41's record); shipped AdamW byte-identical. Spec:
    training-stack.md, "The optimizer factory is CUDA-Adam-specific
    behind a general contract".
50. **Epoch-truth under chunking** (45M-38, 2026-07-12, CRITICAL —
    fourth in the critical-code sequence, after the CMB reopen).
    The loop drops the ragged batch of EVERY loader chunk while the
    resident-encoded branch sizes chunks from bytes with no bs
    rounding (batching.py:359) — load = 2*bs - 1 discards bs - 1
    rows per chunk (near half the epoch), epoch semantics become
    memory-dependent (two GPUs, same seed, different meaning of
    "epoch"), and the EMA accounting proves it executed
    (chunk // bs summed). Contract: placement changes I/O grouping
    never batch count; non-final chunks exact bs multiples; only the
    global n_train % bs tail dropped; steps_per_epoch == n_train//bs
    in every regime, reported and recorded; docstring corrected.
    Spec: training-stack.md, "VRAM chunk boundaries silently change
    the rows used per epoch".
51. **MPS float16 AMP scaling** (45M-39, 2026-07-12; HIGH on the
    Apple/MPS dev path, CUDA bfloat16 unaffected). MPS autocast
    selects float16 (~1702) with a plain backward/step and ZERO
    scaler code repo-wide (untruncated grep) — small gradients
    underflow to exact zero, silently training partial/dead networks
    exactly where zero-init heads and soft-start gates make early
    gradients small; the docstring falsely says bfloat16 (~1560).
    Scaler-or-refuse on MPS; canonical order scale -> backward ->
    unscale -> finite contract -> clip unscaled -> step -> anchor ->
    EMA (extends the 45M-29 order; interlocks units 14/24/41);
    skipped steps advance nothing; resolved dtype + policy
    persisted; docs corrected. MPS acceptance leg on Apple hardware
    (or the teaching-error branch, testable anywhere); CUDA legs
    prove the shared ordering. Spec: training-stack.md, "MPS float16
    AMP has no gradient scaling".
52. **Head padding coordinate truth** (45M-40, 2026-07-12,
    CRITICAL for masked cosmic-shear CNN/TRF heads — fifth in the
    critical sequence). pad_idx substitutes RANK for coordinate
    (counts only; equal-count different-mask bins indistinguishable,
    plain.py:430/ia.py:374) and padding is zero only at the initial
    scatter — conv bias/activation/FiLM write invalid slots and the
    next kernel routes fabricated values into valid positions (the
    two-block adversarial composition); TRF updates the whole token
    rectangle unmasked (ia.py:925); ragged n_tokens included.
    Contract: real coordinate-slot scatter + persisted boolean
    validity mask, reapplied after every block/FiLM; layout-aware
    attention masking; bitwise no-mask preservation; rectangular
    families proven unchanged; pre-map artifacts refused; docs
    corrected. Spec: models-and-designs.md, "Head padding loses the
    coordinate map and fabricates hidden state".
    45M-41 (documentation-only, 2026-07-12) FOLDED THREE WAYS, no
    new unit: the shape-based weight-decay explanations
    (blocks.py:51/:95 teach the abandoned ndim rule; the real rule
    is the role allowlist) -> unit 49; the encode/decode direction
    reversals in the scalar and CMB geometry error rationales
    (scalar.py:127, cmb.py:186 — encode divides, decode multiplies)
    -> the documentation batch (31 + 34, beside 45M-17; "geometry
    totality" as addressed does not exist — recorded deviation); the
    universal-bfloat16 AMP claim -> already in unit 51. Completion
    condition: untruncated zero-stale-hits search. Record:
    conventions-and-workflow.md, "45M-41 fold-ins".

45M round bookkeeping (2026-07-12): 45M-05 RETRACTED by the red team
(ordinary conversion chains accepted; no source-style gate — matches
the standing user ruling). 45M-08 was index-only; Architect verified
it directly (np.savez overwrite, compute_cmb_covariance.py:766-768)
and folded it into unit 3 as the covariance-producer transactional-
publication clause (see artifacts-inference-warmstart.md). 45M-09
(MPS interpolator parse/spline-contract documentation) was named in
the index but its full block arrived only later the same day —
RESOLVED: verified whole and folded into unit 16 as the "45M-09
amendment" (families-background-mps.md); the 45M index is now fully
adjudicated. LATE ADDITION 45M-42 = queue 53 (Optuna journal
experiment identity: no manifest/digest, resume mixes incomparable
science; spec training-stack.md "The Optuna journal has no experiment
identity"; slots in the campaign phase before the docs batch,
interlocks unit 37's implementation identity + the digest machinery).
45M-43 AND 45M-44 RETRACTED by the red team
(2026-07-12), retraction Architect-verified: validate_transfer
(experiment.py:1321-1331) makes transfer single-phase V1 — the
frozen-head state and a head-lr override are unreachable, so unit 54
is WITHDRAWN (the number stays retired, not reused). Audit lesson
recorded in training-stack.md: mechanism verification without a
reachability proof at the validator boundary no longer earns a queue
number. 45M-20 amends unit 22
(training-stack.md, "Selection-record amendment"); 45M-12/13/16/14
carry the red team's priority order and 36 is scheduled first among
them. The three-gate rerun and the
14(a+b+c+d+e) -> 36 -> 42 -> 43 -> 14(g+h) -> 50(+60+14f) -> 52 -> 55 -> 22(+20) -> 13(+01) order
define the active pipeline (updated with the third 45M batch: the CMB
amplitude-law reopen 42+43 slots right after the BAOSN quadrature;
unit 14 gained the 45M-24 safe-sqrt producer increment; unit 15
gained the 45M-26 domain-helper amendment; 44 and 45 queue with
their campaigns).

FIFTH 45M BATCH (2026-07-12, post-retraction, all three
Architect-verified before placement): 45M-45 = unit 7 AMENDED (the
syren alias-consistency boundary: syren_params_from silently prefers
As_1e9 over As and w over w0 with the discarded name unchecked; both
public callers pass complete mappings and the shipped evaluate YAML
defines both amplitude names — reproduced live, conflicts 0.7667 /
0.2449 max rel with every spectrum finite; spec
artifacts-inference-warmstart.md). 45M-46 = NEW UNIT 55
(repeated-training state isolation: transfer.refine mutates the
shared _transfer_base in place — set_live(True) never reset, no
weight/flag restore — and all four repeated-training drivers reuse
one experiment, so sweep/tune/bakeoff/N-train results are order- and
worker-dependent and each point's "pretrained" anchor drifts to the
predecessor's W; REACHABILITY VERIFIED FIRST: validate_transfer
supports refine on cosmic shear (:1368-1410) — the standard the
45M-43 retraction set is satisfied; spec training-stack.md; slots
after 52, before 22). 45M-47 = unit 14 AMENDED, increment (d) (the
epoch reduction overflows float32 before the accumulator: a finite
per-batch loss publishes an Inf epoch loss; host-float64 accumulation
+ an epoch-level finite check; extends the finite-contract gate).
Unit 14 now closes on a+b+c+d + gate.

SIXTH 45M BATCH (2026-07-12, both Architect-verified and
numerically reproduced before placement): 45M-48 = NEW UNIT 56
(generators mark non-finite science payloads successful: no
boundary validates the computed payload before _dv_write and the
failed[i] = False clear — serial :908-921, MPI :990-999/:1048-1057,
all _dv_write overrides blind, only the allocator's first payload
shape-checked, producers close nothing (mps :390 checks only
pre-cast pk_lin); reproduced: float64 [1, NaN, Inf, 1e100] stores
as float32 [1, nan, inf, inf] with failed = False, the 1e100
element finite pre-cast and non-finite in the dump; spec
data-generation-and-cuts.md; joins the file-set/ingress campaign,
cluster now 8+17+25+26+28+33+56; final dataset closure stays with
the wave-1 dataset-readiness unit). 45M-49 = unit 11 AMENDED (the
float32-resolution guard can itself store a zero divisor: the
float64 relative test scale <= 8*eps32*|center| at scalar.py:145 /
grid.py:191 / grid2d.py:196 cannot see absolute underflow —
reproduced: targets [0, 0, nextafter32(0,1)] are accepted and store
float32 center 0.0 / scale 0.0, encode divides by exact zero, and
on grid2d the missed column is neither pinned nor refused;
stored-representation validation contract + covariance sqrt-scale
representability; spec artifacts-inference-warmstart.md; torch
legs join the family geometry identity gates, board-listed).

SEVENTH 45M BATCH (2026-07-12, all four Architect-verified; the
capture mechanism reproduced with the exec'd real function body):
45M-48 ADDENDUM = unit 56 AMENDED (numpy broadcast relabeling:
exact key set + exact predeclared shape, no scalar/length-one
broadcast, cast -> finite -> write once -> read back -> only then
clear the failed bit; family _dv_write never validates; the
duplicate-write rider REJECTED as factually absent — one
assignment at generator_core.py:588-590 on both main and branch).
45M-50 = unit 24 AMENDED (scalar fine-tune saves an artifact with
no finetuned_from/finetune_extra_names though the universal
contract claims every family; one shared provenance assembler;
spec artifacts-inference-warmstart.md). 45M-51 = unit 9 AMENDED
(scalar --diagnostic calls the plain-only local_linear_floor
unconditionally on a legal NPCE run — train, save, then a
deterministic ValueError; capability-owned eligibility; spec
training-stack.md). 45M-52 = NEW UNIT 57 (the generator reads
tmp.read() inside the capture block with no flush of Python/C/
Fortran buffers — reproduced: print() and libc.printf captured
EMPTY with the real body, the C text leaking to the RESTORED
stream, and one block's un-flushed error text landing INSIDE THE
NEXT capture — cross-sample misattribution both directions; a
declared-fatal solver string can be buffered past the guard and
the row marked successful; flush-or-isolate contract, status-API
route satisfies it via unit 33's harmonization; spec
data-generation-and-cuts.md; cluster now 8+17+25+26+28+33+56+57).

EIGHTH 45M BATCH (2026-07-12): 45M-53 + addendum = unit 14
REOPENED, increment (e) — the training-only guard asymmetry (eval
accepts the finite negative chi2 training rejects; a corrupted row
scores perfect and can win best-epoch) plus the compiled gate arm
that greens on any exception. Architect rulings recorded with the
spec: the Implementer's fifth sqrt site CONFIRMED; the absolute
_CHI2_NEG_TOL = 1e-6 SUPERSEDED by the scale-aware band
max(1e-6, 32*eps*n_terms) (per-run constant: compile-safe, not
batch-poisonable); (e) slots BEFORE 42+43. Also this batch:
Implementer closures 14(c) 97963b8, 14(d) 63880d1, 36 387c650 all
recorded, pending batch audit; 14 reopens on (e) only.

NINTH 45M BATCH (2026-07-12): 45M-54 = unit 16 AMENDED (the
mps-smoke range leg passes on ANY exception; the amended legs pin
the vendored check_ranges contract — LoggedError naming the
coordinate, the requested value, and the stored limit,
emul_mps.py:144-165 — on all four boundaries with an in-range
control first; Architect precision ruling: an always-raising P
already reds the gate through the lifecycle leg, so the
discriminating mutant is in-range-healthy + wrong-class refusal —
today's leg greens it, the amended leg must red it). 45M-55 = NEW
UNIT 58 (BAOSN distances integrate from z = 0 while the generator
schema FORBIDS a zero-starting SN grid: comoving_distance_grid
cubic-extrapolates c/H through the untrained [0, zmin), and the
docstring blesses it as legacy-verbatim — a legacy convention now
adjudicated WRONG, the 45M-12 precedent; reproduced with the REAL
module: z_sn=[1,3,600] serves chi(1) +2.06% high, [2,3,600]
+11.26%, extrapolated H finite/positive/monotonic so unit 15's
guards are blind; shipped-scale zmin (0.001 example, 0.01 board
fixture) bias < 1e-3% — the bite is the open schema class;
contract: z_sn[0] == 0.0 exactly, no extrapolating fill in
comoving_distance_grid or distance_interpolators, legacy Hubble
artifacts with z[0] > 0 refused at adapter load with a migration
message, the [0, z_max] window declared only after proving the
persisted first node is zero, cubic + corrected Simpson untouched
on valid grids; fixtures migrate: generator docstring example,
bsn_smoke.py:97, SIX bsn_identity 0.001-starting grids; rides the
fourth wave WITH unit 15, before the EMUL2 acceptance; spec
families-background-mps.md). Also this batch: Implementer closure
42 (5661c08, CMB amplitude-law metric fix) recorded, pending batch
audit. CAPABILITY NOTE: the cocoa clone's python on the Mac dev
box (Cocoa/.local/bin/python) carries torch 2.6.0 (CPU + MPS
backend), cobaya 3.6.2, and scipy 1.12.0 — Mac probes can now
exercise REAL torch/scipy/cobaya paths (the 45M-55 reproduction
imported the real background.py under it); board gate runs stay
workstation-owed.

TENTH 45M BATCH (2026-07-12): 45M-56 = NEW UNIT 59 (top-level
config keys are never censused: every nested block raises on
unknown sub-keys — param_cuts :540, per-head scheduler :282, cmb
:686, grid :833, mps :948 — but set(cfg)/cfg.keys() appears
NOWHERE, untruncated grep; branch selection is
cfg.get("transfer"/"pce") (experiment.py:625/:757/:772) and the
sweep driver reads cfg["sweep"] raw (family_drivers.py:92), so a
misspelled trasnfer:/pec:/seep: block silently builds a DIFFERENT
design and the raw saved YAML claims a feature that never
executed; contract: one explicit top-level schema at from_config +
every driver load boundary before device/staging/source/artifact
work, corpus-derived allowlist audited propose-first,
close-spelling suggestion in the error, executed-composition
statement riding unit 41's resolved record; campaign phase beside
unit 41; spec training-stack.md). 45M-57 = NEW UNIT 60 (the
reported validation "median" is torch.median's LOWER middle sample
for every even n_val — [0,1,9,10] reports 1, the ordinary median
is 5, live-verified; the value drives ReduceLROnPlateau
(training.py:2147/:2202), breaks best-epoch ties at equal frac
(:2163-2166), and is persisted/plotted; FIVE gate files
manufacture their references with the SAME op (cmb_smoke:389,
bsn_smoke:193, mps_smoke:203, finite_contract:137/:873,
ge_c_eval_bs:99) — the board encodes the defect; RULING: adopt the
ordinary 50th percentile (torch.quantile(0.5) verified equal on
both parities), ONE shared reduction for eval_val + scheduler feed
+ tie-break + histories + gate references, the five gate sites
migrate in the same unit, odd-n_val values byte-identical,
even-n_val medians/trajectories change — USER-VISIBLE, declared;
rides WITH unit 50 after 14(e) lands (same eval_val surface); legs
join ge_c_eval_bs (board.py:369, already drives the REAL
eval_val); spec training-stack.md).

ELEVENTH 45M BATCH (2026-07-12): 45M-58 = unit 14 REOPENED on
increment (f) — the landed eval_val validates the chi2 ROWS then
publishes mean = c.mean().item() as a raw float32 reduction
(training.py:1539 at HEAD), violating unit 14's own clause 4
("mean/median/fractions must be finite before the publication").
Reproduced through the REAL eval_val on the Mac cocoa python:
eight finite float32 rows at 1e38 pass the domain predicate,
median publishes finite 9.99999968e+37, and mean publishes INF
(float64 reference: 9.99999968e+37). Distinct from 45M-47 (the
training-epoch loss accumulator); eval_source_chi2 returns per-row
values only, so (f) is eval_val's published reductions: compute in
float64 on the CPU tensor, validate EVERY published scalar/vector
after reduction (raise naming the reduction — no sentinel repair),
ordinary range unchanged to documented ~1e-7 relative tolerance
(histories not byte-identical, USER-VISIBLE declared); gate Part I
in finite-contract with the float32-restoring mutation arm, CPU +
CUDA lanes (backend non-overflow = control, never permission).
(f) rides the SAME eval_val visit as unit 60 — pipeline slot
50(+60+14f). Also this batch: QUEUE 43 RULINGS delivered
(families-scalar-cmb.md): new law NAME as_exp2tau_ref, no parallel
version field, old-law artifacts refused with the retrain error;
as_ref/tau_ref sourced from an explicit validated data.cmb config
pair (required, no default), persisted as resolved float64 state
keys as_ref/tau_ref, _factor reads them with no fallback; staging
verdict line confirmed; retrain confirmed; 43 is GO. Implementer
checkpoint recorded: five self-committed increments this run —
14(c) 97963b8, 14(d) 63880d1, 36 387c650, 42 5661c08, 14(e)
420bce2 — all pending batch audit; unit 14 closed a-e and reopens
here on (f) only.

TWELFTH 45M BATCH (2026-07-12): 45M-60 + addendum = unit 14
REOPENED on increment (g), PREEMPTING queue 43 — the landed (e)
band uses n_terms = w^2 on the dense base (an open, documented
Implementer resolution of the Architect's ambiguous ruling phrase
"count of summed products", whose own parenthetical "n_dv for the
plain whitened form" anchored WIDTH; the ambiguity was the
Architect's, the consequence decides): at production widths the
band is 2.32 (w=780) / 34.33 (w=3000), so a dense chi2 of -2 is
normalized to a PERFECT 0 — the exact false-crowning failure (e)
exists to close — and the shipped Part H tests only the 1e-6 floor
because PoisonChi2 omits _chi2_n_terms (finite_contract.py:913).
RULING: n_terms := the per-row kept WIDTH for EVERY family (band
0.002975 at w=780, 0.011444 at w=3000 — refuses -2/-4 at every
production width; depth-not-count derivation recorded in the
spec); ONE base-class definition, CmbDiagonal override retired,
ScalarChi2 (diagonal, caught inheriting w^2 by the addendum)
correct automatically; metric census + declared-width doubles +
production-surface Part H legs incl. the w*w-restoring mutation
arm and a measured ill-conditioned SPD valid control; band may
only ever WIDEN on measured valid evidence. 45M-59 = NEW UNIT 61
(plot_learning_curves sets log y unconditionally, plotting.py:304,
so a PERFECT fraction of exactly 0 — the best outcome — is dropped
from the published figure, while plot_sweep_curve already
implements the correct linear-on-zero policy :372-373 with the
docstring saying why; contract adopted + ONE shared scale-decision
helper so the two public paths cannot drift; seven CPU legs;
campaign phase; spec training-stack.md). Crossing note: the
Implementer status predates the queue-43 rulings — the law-name
default (as_exp2tau_ref) matches, but the fiducial source is RULED
as the explicit data.cmb config pair (required, persisted
resolved), NOT the covariance-fiducial default the Implementer
proposed to assume; families-scalar-cmb.md "QUEUE 43 RULINGS" is
binding. Implementer real-torch Mac capability acknowledged
(cosmolike-free modules validated as real functions now).

THIRTEENTH 45M BATCH (2026-07-12): 45M-61 = unit 14 REOPENED on
increment (h), the diagnostic score boundary — local_linear_floor
calls chi2fn.chi2 directly (diagnostics.py:226-227) and interprets
the unchecked values immediately (f_floor/f_hard > 0.2 at
:238/:240, median at :241) while ONLY the model arm uses the
guarded eval_source_chi2 (:230-233); _chi2_domain appears NOWHERE
in diagnostics.py (untruncated census); three more direct producer
sites at :414/:621/:745 (cmb/grid/grid2d residual pages).
Reachable today: from_state splats straight into the constructor
with no PSD check (output.py:249, :163-186; unit 11 queued not
landed), so a one-coordinate Cinv=[[-1]] state serves
dchi2_floor = -1 -> f_floor = 0.0 "perfect", median -1 — one
returned record carrying two different definitions of a valid
chi2. Contract: ONE shared public score-domain helper beside
_chi2_domain taking the LOSS OBJECT (family width + compute-dtype
provenance); the floor and the three family producers validate
their OWN vectors before threshold/median/decile/plot/persist;
corruption raises naming diagnostic/producer/count/positions/min/
band; unit 11 = defense in depth not substitute; unit 9's
honest-unavailability must NOT convert corruption to
"unavailable"; valid output byte-identical; seven red legs incl.
the floor-guard-deletion mutation and per-family refusal+identity
controls, in a board-listed diagnostics gate (Implementer proposes
the home). 45M-60 SECOND ADDENDUM = increment (g) AMENDED —
eval_source_chi2 computes chi2 in the loss compute dtype then
derives the band from the UPCAST tensor (training.py:1608
.double(), :1620 band from dchi2_t.dtype): float64 eps collapses
the band to the 1e-6 floor (raw 5.5e-12 at w=780 vs float32
0.002975), so -5e-4 is zeroed by training/eval_val and REFUSED by
the diagnostic — an executed contradiction of the "same predicate
and band" comment (:1609-1615). Amendment adopted whole: band from
the compute dtype captured BEFORE storage casts, validate in
compute dtype then convert for reporting, provenance never
inferred from an upcast, same ordering in (h)'s helper, the
three-boundary one-verdict gate leg, .double()-restoring mutation
arm, genuine-float64 control; rides finite-contract, no new file.
SEQUENCING RELAXED: the (g)-preempts-43 ruling was issued when 43
was believed unstarted; the Implementer status shows 43's loss
side built+verified UNCOMMITTED in losses/cmb.py — the same file
(g) edits — so 43 FINISHES FIRST (avoids a half-unit and a
same-file collision), then 14(g+h) as one visit, then
50(+60+14f). Continue answer to the Implementer: YES, finish 43
as scoped.

FOURTEENTH 45M BATCH (2026-07-12): 45M-62 = NEW UNIT 62 —
validate_grid requires data.grid.units but never checks the VALUE
(probe through the REAL validator: 'bananas', True, None, 3.14 all
accepted; validate_grid2d in the same file refuses by name at
:967-972); GridGeometry str()-coerces and persists; the only value
check in the program is emul_baosn's exact-pair refusal at LOAD —
an expensive successful training run can save a pair its intended
adapter can never serve. Contract: ONE shared quantity->units
registry beside TARGET_LAWS read by BOTH validate_grid and
emul_baosn (the adapter already imports the emulator package),
exact-pair check before staging, no str() coercion (wrong types
are schema errors), constructor+from_state defend the artifact
side, shipped configs byte-identical, errors name received vs
allowed tuples; NARROWED: no per-quantity law restriction
(log_offset is legitimate for either quantity). Six legs incl. the
producer/consumer-agreement mutation arm, in bsn-identity. Rides
the wave-4 background visit WITH 15+58. 45M-63 = NEW UNIT 63 —
D-MP9's constant pin is value-, quantity-, and coordinate-blind
(from_stats pins ANY constant column; probe: pklin/none stale
column -> decode serves 12345.6 for every cosmology; a forged
const_mask on a varying pklin column is accepted by from_state;
the mps_identity leg itself blesses a boost constant of 7.0 under
both laws — neither B=1 nor residual 0). Contract: a pin is legal
only for boost AND center at the law's identity within a
documented float32-derived tolerance AND a per-z contiguous low-k
prefix; a constant pklin column is a loud partial-dead-dump error;
readback validates the mask against the persisted facts in the
constructor; PRECISION RULING: no policy/version field (the
queue-43 no-version precedent — legality recomputes from persisted
facts); the whole-surface guard and the valid boost pin stay
byte-identical; eight legs incl. the quantity-blind-pin mutation
arm, in mps-identity, the 7.0 legs replaced on record. Rides the
wave-4 MPS visit beside 16. Distinct from unit 11
(representability vs permission) — recorded in both specs. No
critical-path change: 43 -> 14(g+h) -> 50(+60+14f) unchanged; both
new units are wave-4 riders. Specs: families-background-mps.md
"UNIT 62 (45M-62...)" and "UNIT 63 (45M-63...)".

FIFTEENTH 45M BATCH (2026-07-12): 45M-64 = UNIT 33 AMENDED — the
background generator's repaired lifecycle is verdict-blind: the
returned LogPosterior is DISCARDED
(dataset_generator_background.py:335), success = absence of three
keywords in captured terminal text, and the provider getters run
unconditionally. Proven on REAL cobaya 3.6.2: a rejecting
component returns logpost = -inf with no raise and no keyword
text while the prior precheck stays finite — a rejected point can
serve the PREVIOUS point's finite H/D_M as a successful row, the
exact stale-physics class cached=False was built to close, and
entry 33's own "worked reference" is the patient. Amendment: the
verdict (finite LogPosterior.logpost, the documented API) is the
acceptance fact before ANY provider read; text scans demoted to
diagnostics; ONE shared acceptance helper across all four
drivers; PRECISION CONCRETIZATION: real cobaya has no provider
generation token — provenance = cached=False + verdict-before-
first-getter (recorded derivation), the fake-Cobaya gate proves
the ORDER mechanically (instrumented fake provider: zero getter
calls on rejection; generation-tagged arrays on acceptance);
seven legs incl. the discard-and-scan mutation arm and the
four-driver census; science dumps generated before the landing
are regenerated after it. 45M-65 = UNIT 11 AMENDED — a valid
one-parameter covariance kills training by dimensional accident:
np.loadtxt returns a 0-d array for a 1x1 covmat and eigh raises
LinAlgError before training (proven through the REAL
from_covmat); np.cov shares the collapse for one feature; the
same probe proved parameter.py is a LIVE instance of unit 11's
mechanism (a) — a negative variance builds sqrt_ev = [nan, 1.0]
silently. Amendment: exact 2-D-square normalization that never
blesses malformed input, name/width/center agreement naming all
three dimensions, finite/symmetric/strictly-PD at all THREE
parameter-side sites (from_covmat :130, np.cov :264,
AmplitudeFactorGeometry.from_covmat :422 — the third site found
in adjudication), multi-param byte-identity; CPU legs + torch
round-trip legs in the board-listed scalar-identity gate. Both
are amendments to QUEUED units — no new numbers, no
critical-path change; 33 stays in the ingress cluster, 11 where
queued. Relay note: the red team's "62 through 64 awaiting"
crossed my fourteenth-batch commit in relay — 62/63 were
adjudicated at 8364598. Specs: data-generation-and-cuts.md
"UNIT 33 AMENDED (45M-64)", artifacts-inference-warmstart.md
"UNIT 11 AMENDED (45M-65)".

SIXTEENTH 45M BATCH (2026-07-12): 45M-66 = UNIT 62 EXTENDED — the
log_offset offset is presence-checked only: the REAL validator
accepted +Inf, NaN, "inf", and True; from_targets' float(offset)
then clears BOTH guards on finite Hubble rows (Inf > 0 passes
positivity; NaN <= Inf is False so the degeneracy guard reports
nothing) and returns center [inf, ...], scale [nan, ...], encode
all-NaN; the poisoned state round-trips from_state. Extension:
explicit finite non-bool real at the validator, from_targets
finite-validates offset/rows/center/scale BEFORE comparative
guards, four separately-named law-domain failures, rebuild
applies the same contract, finite mechanism SHARED with unit 11
(no second mechanism), six legs incl. the
comparative-guards-only mutation arm, bsn-identity home, wave-4
background visit. 45M-67 = UNIT 2 EXTENDED — sigma8 is
advertised unconditionally while _compute_sigma8 relabels the
nearest stored z within 0.01 as z_eval (P(k, 0.009) served as
z = 0, 0.896% deterministic bias; above 0.01 a raw SciPy bounds
error) and integrates the top-hat over whatever k interval the
generator validator admitted (lo < hi, nk >= 8 is the WHOLE
constraint): the exact shipped expression on an admitted 1..10
grid reproduces the red team's numbers digit-for-digit —
0.0000884680 vs reference 0.0049304012, ratio 0.01794, a 98.2%
silent underestimate. Extension: never-relabel (exact stored
z = 0 or refuse naming the range), conditional advertisement,
pre-integration validation, documented k-completeness criterion
tied to the top-hat integrand with certification facts persisted
in the manifest, float64 + radicand validation, shipped
wide-grid preserved subject only to the OPEN R-8-vs-8/h USER
RULING (still owed); eight legs incl. both mutation arms,
numeric legs CPU, the registration + Boltzmann-comparison legs
in mps-identity/mps-smoke. Both are folds into existing units —
no new numbers, no critical-path change. Specs:
families-background-mps.md "UNIT 62 EXTENDED (45M-66)" and
"UNIT 2 EXTENDED (45M-67)".

SEVENTEENTH 45M BATCH (2026-07-12): 45M-68 = UNIT 8 EXTENDED —
load_source verifies the .paramnames sidecar (check_paramnames
:527-533) then discards the resolved names and slices with the
positional default param_cols = slice(2, -1) (:448/:536);
pool_size repeats the literal slice (experiment.py:3303).
Reproduced through the REAL load_source: [weight, lnp, a, b,
d1*, d2*] with a fully-declaring sidecar -> C width 3 vs the
2-name covmat (whitening mismatch); [weight, lnp, a, b] -> C
width 1, sampled b silently dropped; the same file's
_scalar_columns already resolves by name and its docstring says
the fixed slice cannot locate derived columns. One shared
named-column resolver (generalizing _scalar_columns) for
load_source/load_scalar_source/pool_size/checkpoint-reload/
generator-readback, no last-column-is-chi2 inference, exact
uniqueness/presence/width/order before selection, current
generator form byte-identical, missing sidecar = one documented
legacy contract or refusal, seven CPU legs incl. the [:, 2:-1]
mutation arm and the unit-11 one-row composition leg; rides
unit 8 in the ingress cluster. 45M-69 = UNIT 4 EXTENDED — the
board reuses PASS verdicts across executable changes (_passed is
status-only, run_board.py:866-868; records carry status/detail/
ts; HEAD-at-run is written but never read): digest contract
adopted whole (stored per verdict, equality-gated reuse,
legacy = STALE, current-digest dependencies, STALE display,
force-rerun preserved, log/notes commits never invalidate),
digest surface ALIGNED with entry 4's dirty-watch expansion,
propose-first design, eight CPU legs + the status-only mutation
arm; interim rule: --force-rerun stays the manual truth control.
45M-70 = UNIT 33 AMENDED AGAIN — the same verdict blindness
45M-64 proved in the generator exists at all six gate-side
lifecycle sites (cmb_smoke :441-442, bsn_smoke :248-252/:291-295,
mps_smoke :331-335/:376-380) and the CMB covariance producer
(compute_cmb_covariance.py:420-422): LogPosterior discarded,
getters read, no cached=False, so the gates prove readability
not acceptance and two-call forms can bless stale provider
state. Eight clauses adopted; ORDERING RULING: one shared
acceptance definition, whichever side lands first establishes
it (gate legs ride the wave-4 family gate visits, before the
ingress cluster; home importable from gates/ AND
compute_data_vectors/); rejected point = zero getters = red;
CAMB-reference rejection never truth; six workstation legs in
the board-listed smoke gates incl. the instrumented
getter-count and discard mutation arms. All three are folds into
existing units — no new numbers, no critical-path change. Specs:
data-generation-and-cuts.md "UNIT 8 EXTENDED (45M-68)" and
"UNIT 33 AMENDED AGAIN (45M-70)"; gates-and-board.md "UNIT 4
EXTENDED (45M-69)".

ARCHITECT BATCH AUDIT (2026-07-13): everything landed after 05d4937
audited from the merge-candidate tree with the Architect's own gate
reruns (all CPU gates green) + independent code reads. Queue 43 /
14(h) / 60+14(f) PASS their contracts. The red-team wave 45M-71..90
(implemented without adjudication while the Architect was offline) is
ACCEPTED post-hoc; the red-team audit's critical reopens are
independently CONFIRMED — above all the stage_source order defect
(same seed, different minibatches by host-RAM availability; the
stage-ram gate sorts-before-comparing and cannot see it). Repair
queue 1a/1c VERIFIED; queue 4 (c9ace04) + 6a (dce3d69) AUDITED PASS
and cleared to merge. The five evidence-map questions are now BINDING
rulings (red-team answers ratified, plain-leg-name aid scheme, plus
the adopted declared-vs-executed reconciliation clause); the 45M-72
rollout spec is APPROVED. SEQUENCING: queue 3 (staging truth, pinned
contract in gates-and-board.md) FIRST, then the 1b manifest proposal,
then the rollout, then 50 -> 52 -> 55 -> 22(+20) -> 13(+01); unit 50
depends on queue 3's canonical order. Governance: adjudication
protocol resumes; the 45M-69 propose-first deviation is recorded and
self-corrected (queue 1b); the (h)-then-43 flag is closed. Spec:
gates-and-board.md "Architect batch audit of the overnight batch
(2026-07-13, Fable)".

RT-2026-07-13-01 BATCH (2026-07-13): queue-3 landing 2c26c34 AUDITED
PASS (riders honored; stage-ram rerun by the Architect from clean
merged main). RT-IMPL-02 CONFIRMED with a sharper mechanism — _git's
global stdout.strip() breaks _dirty_lines' column parse on the FIRST
porcelain line only, so board_config.json escapes its documented
exclusion exactly when it is the only/first dirty entry (live-proved
both directions); queue 1c REOPENED as 1c-bis (contract in
gates-and-board.md), false-red only, no false-green path. BLOAT-01
CONFIRMED -> queue-4 rider (one _is_finite_real owner = training.py,
import direction already pinned). BLOAT-02 CONFIRMED (the seven _dv_*
ops verbatim across background/cmb/mps; lensing censused) -> NEW
UNIT 64, proposal-first storage engine in generator_core, byte-identity
acceptance, sequenced after the ingress cluster. BLOAT-03 CONFIRMED
(_pick_device verbatim in cmb+scalars, variants elsewhere) -> NEW
UNIT 65, shared adapter mechanics module, no superclass, lands with
the wave-4 typed adapter contract. BLOAT-04 CONFIRMED -> queue-6
rider; completion evidence BINDING as the untruncated 108-line
pattern-family scan + reviewed identifier allowlist. Specs:
gates-and-board.md "RT-2026-07-13-01 adjudication";
data-generation-and-cuts.md "UNIT 64"; families-background-mps.md
"UNIT 65".

1C-BIS + BLOAT-01 PRE-COMMIT AUDIT (2026-07-13, Fable): PASS, GO
given. The uncommitted landing audited in-tree: _WATCH_EXCLUDE +
_watched_paths() one-owner, _git(strip=False) transport with the
head-line misparse documented at the parse site, preflight surface
text derived from the owner; experiment.py's _is_finite_real deleted
and imported from training (the pinned direction). Evidence run by
the Architect: board-selftest 46 legs ALL PASS (incl. the head-line,
transport, one-owner, and stripped-head-line mutation arms); the
live head-line probe through the REAL helpers now excludes a
config-only edit (the exact RT-IMPL-02 failure case); cmb-identity
green after the move. Part J live run stays workstation-owed. Next:
Opus commits (one landing), then the queue-1b manifest PROPOSAL under
the seven pre-answered constraints.

TEXNOTES GAP TRIAGE + SIGMA8 RULING (2026-07-13): all 20 Current-gap
paragraphs in texnotes/emulator_code_guide.tex audited against HEAD.
Sixteen map to already-adjudicated unit specs (map in
conventions-and-workflow.md "Texnotes Current-gap paragraphs"); two
are STALE (watch scope — queue 1c landed; the raw-log resume half of
the digest paragraph — queue 1a landed) and their refresh is
RED-TEAM-owed; two are in flight (queue 1b digest closure,
queue 2 reconciliation); one needed the user and is RESOLVED — sigma8
serves the CONVENTIONAL R = 8 Mpc/h (BSN-curvature precedent; no
legacy rename, no dual serving; rides the wave-4 MPS visit; recorded
in families-background-mps.md). NEW BINDING RULE (ownership by USER
RULE, same day): a landing that changes behavior taught by a
Current-gap paragraph NAMES the paragraph in its notes entry, and
the RED TEAM — the guide's ONLY editor; Architect and Implementer
never touch texnotes/emulator_code_guide.tex — updates it via the
handoff loop (custody rule in conventions-and-workflow.md).
The entry-2 sigma8 caveat above is CLOSED by this ruling.

RT-2026-07-13-02..06 BATCH (2026-07-13): five red-team findings, ALL
CONFIRMED by the Architect (code chains + cocoa-interpreter probes;
the curved-distance numbers reproduce exactly at H0 = 70.0). RT-02
(CPU predictions return storage-sharing views of persistent axes;
ownership is device-dependent) -> NEW UNIT 66,
artifacts-inference-warmstart.md. RT-03 (board children inherit the
shell's $ROOTDIR while the board certifies its own resolved root: sh
copies os.environ verbatim, nothing injects, cocoa.py reads the env)
-> NEW queue 1d (one-owner inject-and-record + red legs),
gates-and-board.md; queue 5 now DEPENDS on 1d. RT-04 (gate_gha_f
captures rc_w and never asserts it — warning-then-crash passes) ->
rider on the 1d landing. RT-05 (CRITICAL wrong science: flat-only
bypassed by a global omk; the producer stores chi as D_M) -> NEW
UNIT 67, rides the wave-4 background visit,
families-background-mps.md. RT-06 (KeyError 'latex' after sampling)
-> NEW UNIT 68 (name-as-label ruling), rides the ingress cluster,
data-generation-and-cuts.md. Sequencing AMENDED: 1b -> 1d(+RT-04)
-> 2 -> 66 -> 5 -> rest of 6 -> 50 -> 52 -> 55 -> 22(+20) ->
13(+01). Spec: gates-and-board.md "RT-2026-07-13-02..06
adjudication".

1B PHASE 1 POST-MERGE AUDIT (2026-07-13, Fable): `5e4fded` PASS —
evidence re-executed from a clean checkout of merged main 060a150
(board-selftest 55 legs ALL PASS, --list rc 0; the shared worktree
holds the in-flight phase-2 edit and was not used). Contract-true on
every delta: whole-AST closure, set fixpoint, both censuses, waiver
table = the delta-1 live instances, mutation arms catch the
uncovered-subprocess and unwaived-dynamic-import cases. Architect
probes found two VALIDATION HOLES (no false evidence — no live gate
declares a manifest): a bare-directory covering root validates while
contributing nothing to the closure, and a typo'd extra root
validates silently -> THREE BINDING PHASE-2 RIDERS (root schema
totality; directory-root expansion with zero-member refusal;
inputs-key resolution), spec in gates-and-board.md "1b phase 1 —
post-merge Architect audit". Phase 2 GO under the riders. Also this
date: the landing-divergence incident diagnosed (merge commits never
flowed back to the branch) and the resync ritual recorded in
conventions-and-workflow.md "Landing-block resync ritual".

1B PHASE 2 POST-MERGE AUDIT (2026-07-13, Fable): `24ed07a` PASS —
board-selftest 67 legs ALL PASS + --list rc 0 re-executed from a
clean checkout of merged main 6f3f54f; all three phase-1 riders
probe-verified live (typo root reds; directory roots expand — the
texnotes probe was an Architect false alarm, make_figures.py is a
real member; empty dir reds; inputs keys resolve-not-exist, the
right cross-machine call); pre-manifest via the REAL _resume_state;
stale members named; resolutions deterministic and sorted;
_manifest_seeds is the one owner behind validation AND digest.
Phase-3 flow APPROVED propose-first (population order + first gate's
roots reviewed before landing); queue-1d placement AMENDED: 1d lands
before the first WORKSTATION rerun of any populated gate; queue 2
blocked until 1b population completes. Spec: gates-and-board.md
"1b phase 2 — post-merge Architect audit".

GUIDE REVIEW (2026-07-13, user-authorized Architect edit):
editorial pass against the user's private review standards (kept
outside this repo) = no changes needed; currency pass = red team had
already folded in everything through today's adjudications; the one stale
area (the digest story, pre-1b) fixed in four passages incl. a
pre-manifest resume-table row; PDF rebuilt clean from the repo root.
Custody returns to the red team. Record:
conventions-and-workflow.md "Guide review of 2026-07-13".

QUEUE 1D + RT-04 AUDIT (2026-07-13, Fable): `28d4207` PASS pre-merge
— 74-leg selftest + --list re-executed from a clean checkout of the
branch tip; the five child-environment legs and both RT-04 legs
verified against the queue-1d contract; one-owner census is itself a
leg; minor pin noted (caller env can layer over the injected ROOTDIR
— refuse a caller-supplied ROOTDIR at the next harness touch).
PHASE-3 PROPOSAL (`5d1b065`) APPROVED with two rulings: the flagged
pair goes through the reviewed waiver table (no carve-out lane), and
the input-side limitation closes at batch 4 (driver gates add
explicit board_config data keys). Also this date: the guide review
landed. Spec: gates-and-board.md "Queue 1d + RT-04 —
pre-merge Architect audit".

20M-01/02 BATCH (2026-07-13, Fable): both red-team findings CONFIRMED.
20M-01 (MPS getters default nonlinear=False while the installed
Cobaya base defaults True — an omitted-argument call silently serves
the linear spectrum; probe: inspect.signature on cobaya 3.6.2) ->
NEW UNIT 69, families-background-mps.md, EMUL2-critical. 20M-02
(cmb_residual_diagnostic drops x_enc at the chi2 call and the loss
falls back to the mutable training stash — validation rows scored
with the last training batch's amplitude factors; [3,3] vs
[12,0.75] verified; cmb-smoke's bar shares the call shape) -> NEW
UNIT 70, families-scalar-cmb.md. Both land as ONE increment parallel
to phase-3 population, before queue 2; gates board-listed,
Mac-validatable. Spec: gates-and-board.md "20M-01/02 adjudication".

20M-03..06 BATCH (2026-07-13, Fable): all four CONFIRMED — the
real-consumer protocol cluster, NEW UNITS 71-74. 71 (emul_cmb
advertises generic Cl but refuses the documented protocol; probe:
BoltzmannBase.get_Cl defaults units='FIRASmuK2', so even the
default-argument call fails; RULING = honor the generic contract,
conversions from persisted facts). 72 (scalar outputs written as
top-level state keys can crash on "derived" and silently corrupt
"params"/"dependency_params"; publication moves into the derived
namespace, atomic assembly, reserved-name refusal). 73 (emul_mps has
NO must_provide — accepts Weyl pairs, out-of-domain z, any k_max;
one-verdict capability helper, the BAOSN law now program-wide). 74
(CRITICAL: fixed cosmology facts are not artifact identity — global
mnu 0.06->0.12 serves an unchanged spectrum while the base is 6.76%
sensitive; ONE persisted fixed-facts block shared with 71 + 67,
compared against the global model at startup). Sequencing: 71+72
join the wave-4 CMB/scalar visits, 73+74 the MPS visit; EMUL2
ACCEPTANCE FORMALLY BLOCKED on 67+69+71+72+73+74. Spec:
gates-and-board.md "20M-03..06 adjudication".

20M-07/08 BATCH (2026-07-13, Fable): both CONFIRMED. 20M-07 (the
Cobaya getters return the live calculation cache by alias — get_Cl,
get_cosmic_shear, get_Pk_grid; a destructive first consumer corrupts
every later one; invisible to a .numpy() census) -> UNIT 66 AMENDED:
ownership surface = every public exit incl. the five adapters'
getters, copies at the getter boundary, alias-aware census, folds in
before the unit lands. 20M-08 (the MPS pair check proves axis
equality only and unions names into manufactured compatibility — an
LCDM pklin + w-carrying boost served the w=-1 surface at w=-0.5,
74.5% off the real base; the "one generator run" comment is false)
-> NEW UNIT 75: pair scientific-domain binding read from the shared
fixed-facts block, refusal before serving, union never creates
compatibility; wave-4 MPS visit. EMUL2 blocklist now
67+69+71+72+73+74+75. Spec: gates-and-board.md "20M-07/08
adjudication".

BATCH-1 POPULATION AUDIT (2026-07-13, Fable): `bb370cf` PASS on Mac
scope pre-merge — 74-leg selftest + --list over the populated BOARD
re-executed from a clean checkout; the three closures strictly
broader (harness hashed; stage-ram hashes its tested surface); the
real reruns (pre-manifest -> republish with persisted members) are
the first queue-5 exhibit, Mac preflight refusing on workstation
facts as designed. family-first re-categorization APPROVED (declare
the driver root; open().read() added to the documented blind-spot
list — file reads are INPUT-side keys, not code roots). Spec:
gates-and-board.md "1b phase-3 batch-1 — pre-merge Architect audit".

20M-09 BATCH (2026-07-13, Fable): CONFIRMED — results.py:683
conflates absent head_act with explicit None and lets the
constructor default rebuild a DIFFERENT architecture that strict
loading cannot see (red-team end-to-end probe: prediction
-1.7615941763 -> -1.0 after deleting one recipe field); the
Architect census adds the :677 outer block_opts-presence lane and
every rebuild-path .get( to the same adjudication. NEW UNIT 76
(recipe schema totality: absence raises, explicit None is a value,
signature-derived key census + injected allowlist, transfer-base
recipes included, complete artifacts byte-identical),
artifacts-inference-warmstart.md; board-listed CPU+Torch legs.
EMUL2 blocklist now 67+69+71..76. Red team continues into the
loss/geometry-algebra and generator completion seams. Spec:
gates-and-board.md "20M-09 adjudication".

20M-10/11/12 BATCH (2026-07-13, Fable): all three CONFIRMED. 20M-10
(CRITICAL: factored physical gain multiplies the CENTER-FREE constant
template and re-adds the center after composition — executed gain off
by -c*r0, zero leverage at GG=center; transfer-identity blind at r=0
parity) -> UNIT 77, one composition owner + old-formula artifact
refusal, blocks factored-transfer science (D-TP9). 20M-11 (CRITICAL:
rebuild never reads the persisted rescale fact — warmstart refuses it,
the public predictor serves finite wrong vectors) -> UNIT 78,
refusal-first ruling ("none" only, all five adapters share it), RIDES
the unit-76 recipe-totality landing; EMUL2 blocklist now
67+69+71..76+78. 20M-12 (roughness leaks to scalar/grid/grid2d NPCE
via inheritance + hasattr family test) -> UNIT 79, explicit family
capability, refusal at validation, CMB path exact. Specs:
artifacts-inference-warmstart.md "UNIT 77"/"UNIT 78";
training-stack.md "UNIT 79"; gates-and-board.md "20M-10/11/12
adjudication".

20M-13 BATCH (2026-07-13, Fable): CONFIRMED — float64 output
geometries (documented + recommended) crash both public
physical-composition paths because PCERatioChi2 and TransferChi2
force the physical truth to float32 and contract directly with the
stored-dtype precision (pce.py:266/:310-311; transfer.py:291/:294);
the whitened route is dtype-aware and fine. Fail-loud, not silent.
NEW UNIT 80 (training-stack.md): one shared physical-contraction
owner casting the residual to the precision tensor's dtype/device at
the boundary; chi2 dtype follows Cinv; float32 byte-identical; lands
in the transfer campaign WITH unit 77 as one algebra increment.
Spec: gates-and-board.md "20M-13 adjudication".

20M-14 BATCH (2026-07-13, Fable): CONFIRMED — the CMB amplitude
law's two roles can alias one input column consistently on both
sides (presence-only validation; two parallel names.index mappings
in configure_law AND staging; persisted geometry never validates the
relationship; witnesses exact: 0.8976275921 / 3.8889e-8). NEW UNIT
81 (families-scalar-cmb.md): distinct strings AND distinct resolved
indices, enforced at config and at readback (same-role artifacts
refused), one shared resolved-role mapping, and the
factor-at-reference invariant evaluated through the resolved roles —
which also closes the Architect's deep-pass minor (the
identity-computed staging banner). Rides the wave-4 CMB visit with
unit 71; readback refusal binding before CMB production training.
Spec: gates-and-board.md "20M-14 adjudication".

UNIT 69 AUDIT + SEQUENCING (2026-07-13, Fable): `371b0bd` AUDITED
PASS on Mac scope (all five legs re-executed green from a clean
checkout; explicit branches untouched; leg 5 workstation-owed; the
one red mps-identity leg — bounded-staging streamed mean — fails
identically at the parent and is owned by the binding bounded-grid2d
amendment, NOT unit 69). SEQUENCING RULING for the 20M backlog:
70 next -> 79 -> the 77+80 algebra increment (one landing) -> the
76+78 artifact-totality increment (Mac-clears two EMUL2 blockers) ->
the FIXED-FACTS BLOCK PROPOSAL for Architect review (gates the
adapter cluster 66-amended/71-75/81 inside the wave-4 visits);
population batches 2-5 interleave as review-blocked filler; queue 2
after population; queue 5 carries the accumulated workstation legs.
Spec: gates-and-board.md "Unit 69 post-merge audit + the 20M-backlog
sequencing ruling".

20M-15/16 BATCH + 14-ADDENDUM + PDF RULING (2026-07-13, Fable): all
adjudicated. 20M-15 CONFIRMED (structural-only checkpoint loaders
trust false success bits; corrupt rows resume as finished science)
-> UNIT 56 checkpoint-ingress amendment: resumed success rows
revalidate through the publication predicate, inconsistency refuses
touching neither file. 20M-16 CONFIRMED (fresh writes %.9e float64
chain but computes at an independent float32 cast; append reloads —
one-ULP row-identity mismatch, ~700/1e6 draws) -> NEW UNIT 82 (row
authenticity): one canonical table, bitwise producer == staged ==
loader-recovered on every path. 20M-14 ADDENDUM accepted -> UNIT 81
AMENDED (semantic roles: persisted role registry + the
fiducial-unity check through resolved roles; swapped-role witness
3.4907e-8 re-derived exactly). TRACKED-PDF RULING ratified -> NEW
UNIT 83 (queue-6 doc lane): PDF stays tracked, bound by a
build-manifest gate (digests over the include/graphics closure +
the PDF; no mtimes; manifest written only after a successful
build). CLAUDE.md writer split aligned with the USER RULE
(Implementer writes no notes/ or texnotes/; everything in the
handoff block; 0040bc5 correct). Specs: gates-and-board.md
"20M-15/16 + 20M-14 addendum + the tracked-PDF ruling".

20M-17/18 BATCH + 13-ADDENDUM (2026-07-13, Fable): all adjudicated.
13-addendum ACCEPTED -> UNIT 80 AMENDED (float64 geometries also
crash all four structured heads at y @ W_fd — head basis buffers
cast at their owned boundary; one end-to-end precision owner).
20M-17 CONFIRMED (artifacts persist NO physical input domain; the
tanh witness serves finite answers 23.84%/90% wrong; _as_row
unchecked) -> NEW UNIT 84: persisted per-name support from the
declared prior contract (never observed minima), save+rebuild
validation, startup prior-subset proof, per-point refusal before
encode, no clamping, intersection serving, narrowing-only
propagation, legacy refusal; enforced centrally in
EmulatorPredictor. 20M-18 CONFIRMED (sampled w0pwa silently
degrades to wa=0 in the served syren base while the correction was
generated at wa=0.5; 12.987% miss) -> NEW UNIT 85: one canonical
dark-energy resolver shared by generator and adapter, persisted
parameterization identity in the fixed-facts block, refusal on
underdetermined mappings. Open EMUL2 blocklist: 67, 71-76, 78, 84,
85. Specs: gates-and-board.md "20M-13 addendum + 20M-17/18
adjudication".

20M-18 ADDENDUM BATCH + OWNERSHIP CORRECTION (2026-07-13, Fable):
the wa defect is PUBLICLY reachable — the shipped EMUL2 YAML samples
w0pwa with the wa bridge and masks it only because its one
evaluation point sets w0pwa == w; full-grid magnitudes up to 121.5%
(3.0239% at unit blend weight = the final served error). UNIT 85
AMENDED (spy-gate leg; wa-under-w0pwa = dynamic per-point fact under
unit 74's mechanism, never pinned). RECORD CORRECTION executed on
the unit-7 "full calculate(**params) mapping" sentence (Cobaya
routes only required/supported inputs). USER CORRECTION on notes
ownership: the Implementer MAY edit notes/*.md (resume state); ONLY
texnotes/ TeX sources are red-team-only; CLAUDE.md restored; 0040bc5
was a good-faith over-correction. Spec: gates-and-board.md
"20M-18 addendum + the notes-ownership correction".

20M-18 CORRECTION + 20M-19 (2026-07-13, Fable): the red team
self-corrected 18's reachability — the shipped drop:true YAML is
STARTUP-RED for w0pwa-storing artifacts (a separate defect now in
unit 85's scope: repair the example or refuse with migration text;
it is not working EMUL2 routing today), while the silent wa=0 lane
is the valid NON-drop configuration, proven through a COMPLETE real
Cobaya model (theory assigned the seven names; logposterior served
the wa=0 base at 0.1298745470923877 max rel error vs the requested
wa=0.5). UNIT 85 stands with the corrected record + both-branch
gate. 20M-19 CONFIRMED (const_mask consumed by NO loss — chi2 0 vs
10000 for bit-identical served predictions; the program optimizes a
different function than it serves) -> NEW UNIT 86: one
effective-residual owner masks before every diagonal reduction, zero
residual AND gradient on pinned coordinates, unit 63 stays the
legality authority. Spec: gates-and-board.md "20M-18 reachability
correction + final proof + 20M-19 adjudication".

UNIT 70 AUDIT (2026-07-13, Fable): `5d22634` PASS pre-merge —
cmb-identity 78 legs ALL PASS from a clean checkout ([3,3] and
[12,0.75] verbatim; stash-invariant; wrong-length no-crash) +
diagnostics-domain green; all three diagnostic families pass the
batch's own x_enc; the smoke bar uses explicit params. Beyond
contract: loss() clears the stash in a finally, making stash privacy
structural. 20M-01 and 20M-02 are now both CLOSED on the Mac; their
workstation legs (69 leg 5, the smoke mutation arm) ride queue 5.
Next per the sequencing ruling: unit 79. Spec: gates-and-board.md
"Unit 70 — pre-merge Architect audit".

20M-18 COUPLING + 45M-81 EXECUTED FAILURE (2026-07-13, Fable): the
red team's 18 rejection notice crossed with the already-committed
correction; its ONE new clause is adopted — the shipped-YAML repair
and the canonical resolver are ONE landing (removing drop: true
alone converts loud failure into silent wrong physics). 45M-81 is
now a CONFIRMED CURRENT FAILURE, Architect-re-executed (append with
the same seed restarts default_rng and duplicates 200/200 rows at
the public minimum; no RNG state in checkpoints; the header implies
seed-only provenance): binding amendment in
data-generation-and-cuts.md — RNG state is checkpoint state,
persisted transactionally, restored-or-verified before any append
draw, emcee state included (the queue-5 sampler._random rider joins
it), refusal before touching the old dataset; NO production append
until it lands; unit 82 is orthogonal. Spec: gates-and-board.md
"20M-18 coupling clause + 45M-81 executed failure".

20M-21 BATCH (2026-07-13, Fable): CONFIRMED — the generator writes
raw lnp into GetDist's reserved minus-log-posterior column on fresh
AND append (Architect live probe: GetDist selects the logpost=-10
row over -1 as best fit; ranking/shading/cooling all reverse);
uniform mode fabricates lnp=1/chi2*=-2. NEW UNIT 87
(data-generation-and-cuts.md): minus_logpost materialized at the
publication boundary into column 2, exact chi2*/2 relation, honest
headers, explicit uniform unavailable status, legacy chains
marker-or-refuse. Joins the generator publication cluster (45M-81
RNG amendment + 68 + 82); production Gaussian/MCMC generation
blocked on the cluster. Spec: gates-and-board.md "20M-21
adjudication".

20M-22/23 BATCH (2026-07-13, Fable): both CONFIRMED. 20M-22 (the CMB
adapter publishes finite but impossible spectra — negative TT/EE/pp
and a non-PSD TT/TE/EE triplet pass the shape/finite boundary
through the real lifecycle) -> UNIT 21 AMENDED: auto spectra
nonnegative (never clipped), TE signed, the joint PSD bound within a
storage-arithmetic-only rounding band, no partial state, proof in
cmb-identity. 20M-23 (--append 1 --loadchk 0 is accepted and
DESTROYS the prior dataset — the relation lives in the help text and
is enforced nowhere; live sentinel destruction re-verified in code)
-> UNIT 8 AMENDED: the run-control state machine (0/0, 1/0, 1/1
exhaustive) validates before ANY path mutation, teaching error,
byte-preserving rejection, RNG continuation independent; joins the
generator publication cluster. Spec: gates-and-board.md "20M-22/23
adjudication".

BATCH-2/3 POPULATION AUDIT (2026-07-13, Fable): `2674c23` + `724bade`
PASS pre-merge — 74-leg selftest + populated --list from a clean
checkout; the batch-2 inputs=() deviation RATIFIED as the narrower
truth (Architect census: zero yaml/config/rootdir usage in the three
bodies AND check scripts — the proposal's "with a config YAML"
description was wrong about them); all eight batch-3 bodies clean,
designs+losses roots per the approved exemplar. 14 gates populated;
board reruns = queue-5 first exhibit. Spec: gates-and-board.md
"1b phase-3 batches 2 + 3 — pre-merge Architect audit".

20M-24 BATCH (2026-07-13, Fable): CONFIRMED — --gpu-pack lanes stage
full train+validation data in setup_fn BEFORE any token acquisition
(live instrumentation: 4 simultaneous charged setups vs 1 execution
under four-token exclusive jobs; the estimator charges exactly the
term setup allocates outside the gate). NEW UNIT 88
(training-stack.md): tokens precede job-sized allocations, exact
concurrency arithmetic (1/2/4 for 4/2/1 tokens, setup+execution
combined), permanent-per-lane state accounted times lane count,
honest banner, N-train path audited identically; CPU counter-fake +
workstation tight-fixture legs; blocks --gpu-pack production sweeps.
Spec: gates-and-board.md "20M-24 adjudication".

BATCH-4/5 RULINGS (2026-07-13, Fable): the geo-paths non-existence
test reworks to importlib.util.find_spec — APPROVED and the batch-5
taxonomy refined (waiver-with-roots | rework-to-static |
find_spec-for-non-existence; no fourth category; the retired-module
red stays); batch-4 data keys = a gate_data block parallel to
gate_configs (dotted gate_data.<gate-id>.<label>, resolved by the
existing ladder, resolve-not-exist semantics), with the paste-ready
shape in the ruling. Mixed-commit record note: the Implementer's
resume block entered history in d852b1a and left in 5d46e8d — both
mixed-author; content intact; the diff-staged-hunks rule re-affirmed
for both sessions. Spec: gates-and-board.md "Batch 4/5 design calls
— RULED".

20M-25 BATCH (2026-07-13, Fable): CONFIRMED — the null roughness
path clears nothing while sweeps reuse one experiment + loss per
lane: an enabled->disabled sweep's second point optimizes the first
point's penalty (13.4511995 vs fresh 7.0). NEW UNIT 89
(training-stack.md): complete loss-object state established on every
invocation (absence = CLEAR), one owner, fresh-object equivalence
across order/lane permutations, failed points leave nothing, the
resolved record describes installed state, and the configure_*
census (law, rescaling, transfer roughness) audited under the same
discipline. Blocks production hyperparameter sweeps over loss
blocks. Spec: gates-and-board.md "20M-25 adjudication".

BATCH-5A AUDIT + POPULATION GO (2026-07-13, Fable): `fddf05f` PASS —
geo-paths rc 0 with the live find_spec probe, 74-leg selftest,
15-gate --list, and the declaration proven REQUIRED (the rootless
closure reaches both waived files). Batches 4 (gate_data schema as
ruled) and 5b (cli-strict reviewed waiver entries) are GO; their
landing completes the 1b population and unblocks queue 2. Spec:
gates-and-board.md "Batch 5a (geo-paths) — pre-merge Architect
audit".

DIDACTICS-01/02 BATCH (2026-07-13, Fable): both ACCEPTED into queue
6. 02 (navigation truth: the __init__ "six files" claim vs 27 real
files; the README prose count inside the sentence warning against
prose counts; the missing beginner walkthrough; rollout history in
the Manifest/Gate docstrings) lands FIRST, with a board-selftest
set-equality leg validating the __init__ index + README table
against BOARD, and the docstring cleanup after batches 4/5b. 01 (the
file-by-file teaching campaign: four-paragraph contracts on every
scientific leg owner, test-double honesty, taught Python mechanics,
named constants, shown mutation arithmetic, one Cobaya lifecycle
lesson) lands per gate family under the BINDING mechanical
acceptance: AST docstring census, the untruncated history scan
(merged with the queue-6 108-line pattern family, gates/ explicit),
compileall + selftest, and documentation-only AST equivalence.
Doc-only interleave; nothing preempted. Spec: gates-and-board.md
"DIDACTICS-01/02 adjudication".

DIDACTICS-03..09 BATCH (2026-07-13, Fable): the per-family teaching
batches (03 family gates, 04 staging + memory mechanics, 06 harness
mechanics, 07 torch/numerics) RATIFIED under the binding mechanical
acceptance, with ONE named executable exception (04's coordinate
table gains a stage-ram leg). 05 CONFIRMED + SHARPENED (the BSN
"closed-form" reference IMPORTS the production cumulative_simpson —
zero catch power against it): the honest narrowing + the README
all-family rewording ride the first queue-6 landing; the
independence upgrade = NEW UNIT 90 (independent quadrature reference
+ Simpson-weights mutation arm, board-listed, rides the BSN visit).
08 CONFIRMED (batching.py claims universal global rows — true only
disk-backed post-queue-3; separate filenames do not prove no
leakage): immediate factual correction with the two-regime table +
the assumption sentence until unit 28's disjointness check. 09
RATIFIED (plain-language vocabulary; machine identifiers and the
"leg" term survive via the reviewed allowlist + README definitions).
Spec: gates-and-board.md "DIDACTICS-03..09 adjudication".

DIDACTICS-10..19 BATCH (2026-07-13, Fable): the inference/adapter
didactics wave, all ratified in THREE LANES. Lane 1 factual
corrections (verified: "the h5 alone" vs the required .emul; the
false same-dtype promise vs the NumPy float64 coercion; "six"
syren arguments vs seven; "re-derives no physics" vs owned
composition/sigma8/distances) land with the queue-6 factual bundle.
Lane 2 = NEW UNIT 91, the documentation-examples gate: ONE
board-listed check module holds every executable worked example
(dtype controls, unit tables, the five-point Simpson, the
instrumented Cobaya lifecycle, interpolator shapes, scatter/mixed
fixtures; 04's table leg moves here), keeping production files
strictly AST-identical. Lane 3: the family teaching (15-19, 13)
rides the wave-4 protocol visits so prose describes final behavior;
10b ownership rides unit 66; 12's keyword-arg conversion rides unit
85. BLOAT-04 is subsumed by 15/16. Spec: gates-and-board.md
"DIDACTICS-10..19 adjudication".

DIDACTICS-20..26 BATCH (2026-07-13, Fable): all ratified with four
rulings — the board.py registry rewrite rides QUEUE 2 (one pass,
non-executed evidence labeled UNAVAILABLE per binding ruling 6);
beta stays UNCONSTRAINED (prose describes the executed model; tail
tests to unit 91); power-activation bounds get option-(a) validation
riding UNIT 40; the five-site mkdtemp leak is a behavioral fix in
the 25 batch. Campaign-wide adoptions: the quantifier discipline
(mechanical proof adjacent to every bitwise/exactly-once/all claim)
and scope-and-blind-spot labels on every structural scan ("census"
reserved for complete surfaces). Factual-bundle additions: board.py
:634/:1370, transfer_identity's stale "refine not yet implemented",
the HDF5-typed-attributes correction, geo_paths' stale import-raises
prose (which survived the Architect's own fddf05f audit — gate
audits now read module preambles against executed mechanisms).
Spec: gates-and-board.md "DIDACTICS-20..26 adjudication".

DIDACTICS-27..32 BATCH (2026-07-13, Fable): the headline is 31,
CONFIRMED as a gate-fixture CORRECTNESS defect — all three
real-generator smokes build train AND val with --seed 1234 +
split_seed 0, so validation overlap is 100% and the smoke
validation numbers measure memorization; the experiment.py
"independent selection" comment is false. UNIT 28 AMENDED (distinct
recorded seeds or proven disjoint partition, zero-overlap refusal
before training, alignment proof, same-seed mutation arm); the
fixture repair lands BEFORE queue 5. Others: 27 wrapper-child
reconciliation extends the queue-2 registry rewrite (logged !=
executed; UNAVAILABLE per ruling 6); 28 subprocess ten-field
contract + both-stream failure tails + board_selftest's 4 mkdtemp
sites join the 25 cleanup (nine sites, one landing); 29 triangle
gate STRENGTHENED (panel-specific RGBA evidence + wrong-panel
mutation) and logscan renamed to selected/normalized text equality
("byte-identical" reserved for raw bytes); 30 weight-decay claim
narrowed to decay-term inactivity + BerHu units corrected; 32
RULING option 1 — full labeled child streams persisted into the
gate log, explicit no-timeout statement with rationale. Spec:
gates-and-board.md "DIDACTICS-27..32 adjudication".

DIDACTICS-33 BATCH (2026-07-13, Fable): CONFIRMED and wider than
filed (the universal global-row claim also at training.py:1788 and
:1863) — a rider on the 08 correction, ONE landing for batching.py +
training.py with the shared two-regime table. Ruling beyond the
label: validation errors NAME THE ORIGINAL DUMP ROW in both regimes
(resident staging already holds the mapping — zero new persistence);
torch.cat allocation described separately from the conditional
.cpu() transfer. 28..32 re-filed and already adjudicated. Spec:
gates-and-board.md "DIDACTICS-33 adjudication".

DIDACTICS-34..41 BATCH (2026-07-13, Fable): eight model/loss/
geometry documentation falsehoods, ALL CONFIRMED (ratio + growth
algebra re-verified; the CMB decode direction consistent with the
queue-43 law). Architect widenings: 34's reversed law-direction
phrase has two uncited siblings in inference.py (:180, :532); 40's
multiline escape upgrades EVERY campaign stale-phrase census to
multiline/semantic-capable scans. All routed to the model/loss/
geometry family batches; worked examples to unit 91. Spec:
gates-and-board.md "DIDACTICS-34..41 adjudication".

DIDACTICS-42..57 BATCH (2026-07-13, Fable): sixteen items. FIVE
gate-truth defects (46 golden byte-identity greens on empty
selections — live-reproduced by the Architect through the real
helper; 47 mps-smoke's vacuous filtered .all() passes an all-zero
dump; 48 production-diagnostic never checks the PDF exists; 52 the
OR-joined absence census — the Architect's own deep-pass rider 3,
deferral upgraded to NOW; 53 six-block claim, TT-only checks —
RULING: extend, not narrow) become ONE gate-truth repair batch
BEFORE queue 5. UNIT 92 born (device-audit totality: recursive
traversal, raise-regardless-of-silent, CUDA lane workstation).
Factual bundle grows (42 package intro, 43 set_device + interleave,
49 the .h5/.emul REVERSAL + stage_train claim, 50 factored width).
51's signature-vs-Arguments AST census joins the standing
acceptance; 45 merged into DIDACTICS-11; the 40-addendum extends the
direction census to exception strings. Spec: gates-and-board.md
"DIDACTICS-42..57 adjudication".

DIDACTICS-58 (2026-07-13, Fable): CONFIRMED (census exact: 24 files,
both patterns, full intersection) — the canonical seven-mechanics
harness-Python preamble rides the DIDACTICS-02 __init__ landing (one
file, one edit); no clever framework, per the C-coder rule. Spec:
gates-and-board.md "DIDACTICS-58 adjudication".

DIDACTICS-62/63 (2026-07-13, Fable): CONFIRMED — the banner-as-truth
class (five behavioral-claim gates checking only banners/exit codes;
param-window-cuts accepting any "used n of m" line unparsed). Point
7 adopted as doctrine beside ruling 6: reconciliation proves
execution, never observation. RULINGS: prose narrowed NOW (factual
bundle; queue-2 aids mint from narrowed claims); 63's independent-
reference repair joins the gate-truth batch; 62's parse-and-assert
strengthening = gate-truth increment 2 pre-queue-5, with
joint-training's driver-side phase-boundary digest riding the
training-truth trio if timing is tight. Spec: gates-and-board.md
"DIDACTICS-62/63 adjudication".

DIDACTICS-59/60 (2026-07-13, Fable): CONFIRMED — 59 the
copied-reference gate (ge_c_eval_bs certifies its own chi2 loop, not
the real eval_val; ruling: exercise the real entry point, reuse the
existing production per-row diagnostics surface, no duplicated
formula) and 60 the stale scalar-smoke baseline (0.455 removed with
the causal smoothness claim; baseline recomputed from exact staged
rows + dead-network mutation) — both join gate-truth increment 2,
with 60 sequenced AFTER the unit-28 disjointness repair because its
recomputed number was measured on the overlapping fixture. Spec:
gates-and-board.md "DIDACTICS-59/60 adjudication".

DIDACTICS-61 + 64..71 BATCH (2026-07-13, Fable): the red team's
durable register (notes/red-team-audit-and-didactics-2026-07-13.md,
42--71) adopted — 42-60/62-63 copies reconciled, MATCH. Nine new
confirmations: 61 (a two-column fixture called one-wide; decreasing
already enforces two points but its docstring says non-empty, and a
+Inf first value passes -> finiteness joins the gate-truth batch);
64 (the compile-mode drift arm patches a global that rebuild
provably never reads — arm removed in increment 2; persistence proof
= NEW UNIT 93, CUDA instrumented lane, workstation-owed); 65 merges
into 29 (exact panel-identity acceptance); 66 (self-defined
masked-zero proof, vacuous when dest_idx covers all — gate-truth
batch with 47); 67+68 = one warmstart visit (teach the double .h5
open + define .ia; finite guards added to BOTH perturbed parity arms
— a perturbation-born NaN is currently mislabeled "extras leaked");
69 (exactly six "equally hard" sites -> one canonical
no-equal-learnability sentence, multiline census); 70 (results.py
contradicts its executed CPU normalization; CPU-residency leg added
in increment 2); 71 (the "bit for bit" headline split into the two
real invariants: _PARITY_TOL function reproduction vs bitwise
extras-independence). The 46 golden-leg addendum absorbed (both rcs
discarded at board.py's _golden_leg — rc==0 + nonempty count now
required). Durable-record rule adopted both ways. Gauntlet
unchanged. Spec: gates-and-board.md "DIDACTICS-61 + 64..71
adjudication".

CONSOLIDATED DIDACTICS HANDOFF (2026-07-13, Fable): the whole
campaign (01--71) issued to the Implementer as ONE ordered plan —
D1 navigation truth (02+58), D2 factual bundle (falsehoods jump the
queue), D3 unit-28 fixture repair, D4 gate-truth batch, D5
increment 2, D6 hygiene (nine mkdtemp sites + triangle 29/65), D7
unit 91, D8 lane-3 family visits riding protocol units, D9 queue-2
riders, D10 minted units 90-93 — under the binding per-landing
acceptance (AST censuses, untruncated scans, quantifier discipline,
the didactics voice note read first). The spine is unchanged:
4/5b -> D3 -> D4 -> D5 -> queue 5. Spec: gates-and-board.md "The
consolidated DIDACTICS execution handoff".

REGISTRY-PERSISTENCE AUDIT (2026-07-13, Fable): the red team's
retroactive durable-record pass (codex/architect-docs-static-audit,
57dfe26 notes-only) audited and GO'd — tombstones 45M-10/18 +
20M-20 exist only as explicit not-issued lines (untruncated grep);
retractions 45M-05/43/44 stay visible, not rewritten; the
readiness contract's live-status claims (exit(0) tail, boundary
rewrite, no fail-file consumer) re-verified at HEAD; MEMORY.md's
deletions remove stale numeric snapshots per the DIDACTICS-02
doctrine and demote this ledger to chronology-not-queue (current
sequencing = the consolidated handoff). Landing order: codex branch
(ff to 35f0137), then amazing-keller (2f25450), then resync. Spec:
gates-and-board.md "Registry-persistence audit".

ALIAS CLOSURE (2026-07-13, Fable): d00358c GO'd — the 20M-15 /
unit-56 checkpoint-ingress labels joined at both registry sites;
MEMORY.md names the D1-D10 consolidated handoff as sequencing
authority; the "main dirty" hold was stale (merge concluded as
7743fc6); direct merge ruled safe (ancestor base, disjoint files).
Spec: gates-and-board.md "Registry alias closure (d00358c)".

BATCH-4/5B CHECKPOINT + DEPLOY_DATA SHAPE (2026-07-13, Fable): the
three leftover no-schema declarations audited pre-commit (gwd-c's
empty manifest is reasoned: subprocess-only gate, in-process closure
never reaches the waived imports; gsv/gct rebuild in-process and
correctly declare designs+losses); 18 declared, ok=True. RULING:
ONE shared deploy_data block (semantic logical-key -> rootdir-
relative path), lane chosen by provenance not consumer count,
dotted deploy_data.<key> references, absent-key startup red, no
duplicated literal paths; full derived block returns as a
paste-ready proposal for sign-off. Spec: gates-and-board.md
"Batch-4/5b checkpoint audit + the deploy_data shape ruling".

### Continued red-team findings — ADJUDICATED (Fable, at the merge)

Every item below is verified and placed; none opens a new queue
number: sweep completion truth MERGED with unit 10 (one
sweep-worker-truth unit; anchors spot-confirmed); generator
finiteness FOLDED into unit 17 (the ingress validator's finiteness
extension; int()/bool() coercion + the isinf(NaN) prior hole both
confirmed); nested path + validation axes were ALREADY units 25 + 26
(same findings, adjudicated from the direct handoffs before this
recording merged); the bounded-grid2d amendment is BINDING on the
in-flight revision (both halves — the cancellation was independently
confirmed, the float32-payload clause is new and accepted); the
terminology cleanup rides the same revision as its prose rider. The
red-team index below stands as their record.

These are evidenced contracts, not numbered/accepted queue units yet;
their detailed specs are in the existing topic notes (no new files):

- **Sweep completion truth** (`training-stack.md`): serial and parallel
  turn identical point failures into different final outcomes; NaN rows
  can be published as a normal successful sweep.
- **Generator scalar/grid finiteness** (`data-generation-and-cuts.md`):
  lossy int/bool coercions change requested grids and switches; NaN
  extrapolation and NaN prior results evade comparison-only guards.
- **Nested path + validation-axis identity**
  (`data-generation-and-cuts.md`): Cocoa resolves primary dumps but not
  family covariance/grid/base sidecars; validation axes are inferred
  from training axes and same-width swaps are undetectable.
- **Bounded-grid2d amendment** (`data-generation-and-cuts.md`): the
  in-flight memory fix must use stable Chan/Welford moments over the
  exact stored float32 rows; its reviewed sum/sum-of-squares draft can
  mis-scale or falsely pin a varying column.
- **Gate terminology cleanup**: human-facing board/check comments,
  docstrings, maps, and report labels say "independent known-answer
  calculation/check"; only existing identifiers may retain `oracle`.

Role/sequencing: Fable decides queue numbers and priority. The grid2d
numeric amendment is blocking feedback on the already in-flight unit,
not a later cleanup. The other findings should fold into the existing
train-argument, generator-ingress/file-set, and documentation contracts
where their roots already live.

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
