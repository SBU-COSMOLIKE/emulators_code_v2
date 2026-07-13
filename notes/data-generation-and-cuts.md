# Data generation, staging, and the physical parameter cuts

Consolidated 2026-07-11 from compute-data-vectors-import.md,
param-cuts-nested-block.md, omegam2h2-window-cut.md,
omegamh2-ns-product-cuts.md, triangle-cut-shading-all-windows.md,
data-staging-ram-and-source-dict.md (retired; full texts in git
history). Code homes: compute_data_vectors/, emulator/data_staging.py.
The user-facing story is README section 18 ("Generating the training
set" — the four-generator table + the shared-core walkthrough).

## The generators (compute_data_vectors/)

- One shared core, `generator_core.py` (D-CM3-A): the CLI (15 flags),
  the tempered/uniform sampling, the MPI master/worker farm, the
  checkpointing, capture_native_output. Four thin drivers subclass it:
  dataset_generator_lensing.py (cosmolike dv), _cmb.py (four spectra
  files _tt/_te/_ee/_pp), _background.py (the _h/_dm pair + _z.npy
  sidecars; desert-overlap loud), _mps.py (pklin/boost surfaces + the
  syren _base files when write_syren_base + _z/_k sidecars; the
  wants-Cl CAMB quirk kept verbatim). Fifth tool:
  compute_cmb_covariance.py (the analytic per-multipole covariance;
  families-scalar-cmb.md has its physics; its params block is PLAIN
  NUMBERS with the script's OWN names omegabh2/omegach2 — mirror
  example_yamls/cmb_covariance_lcdm.yaml, both conventions are
  validated loudly and each was re-invented wrong once by a gate
  fixture).
- TWO evaluation idioms, deliberately unharmonized (board runs 1-5,
  2026-07-11): lensing/cmb/mps evaluate each sample through the
  legacy hand-rolled check_cache_and_compute(cached=True) component
  loop — proven by their gates (dumps vary, trainings collapse). The
  BACKGROUND generator instead uses the standard
  model.logposterior(point, cached=False) lifecycle: with its
  background-only requirement set the legacy loop served a STALE
  first-sample CAMBdata (every dump row one cosmology, bitwise), and
  the wants-Cl quirk did NOT cure it (the bsn-smoke dump-variance
  tripwire falsified that hypothesis at spread exactly 0.0). Do not
  harmonize either direction without gate evidence; the full saga is
  in families-background-mps.md.
- The MPS Pk requirement's k_max is DERIVED, not the legacy constant:
  max(2 x k grid top, 20) — equal to the verbatim 200 on the
  production grid (k top 100), ~10x cheaper for a small-grid smoke
  (the first full mps-smoke run burned ~1 h computing k = 200
  transfers against a grid topping at 10). Every dumped k stays
  computed, never extrapolated; extrap_kmax remains the served
  interpolator's tail edge.
- The lensing generator was imported VERBATIM from the user's
  production copy (byte-identical except one header path) —
  battle-tested production code gets NO house-style retrofit; the
  README, not the script, carries the didactic layer.
- Two sampling modes, peers: tempered (--unif 0) samples
  log p_T = [-(1/2)(theta-theta_0)^T Sigma~^-1 (theta-theta_0)
  + log pi(theta)] / T with Sigma~ = the params covmat, correlations
  CLIPPED to |corr| <= maxcorr (default 0.15); uniform (--unif 1)
  fills the temperature-stretched box (bounds widened T*width/5; lnp
  = 1; files tag `_<probe>_unifs`). WHY each knob: T widens the cloud
  past the posterior (chains at the edge must stay inside the training
  support); maxcorr fattens the Fisher pancake PERPENDICULAR to
  degeneracies; --boundary < 1 shrinks val/test INSIDE the training
  support (accuracy dies at the cloud edge).
- The output contract (closes every loop):
  `<paramfile>_<probe>_<T>.1.txt` (weights/lnp/params/chi2*; staging
  drops the bookends), `.paramnames` (first column = cobaya names ->
  ParamGeometry -> h5 -> get_requirements — THE naming loop),
  `.covmat` (the INPUT whitening basis), `.ranges`, the dv store
  (per-family file set), `<failfile>` (recompute with --loadchk 1).
- Loud disambiguation: the generator YAML is a COBAYA yaml with its
  OWN train_args (probe/ord/fiducial/params_covmat_file) — unrelated
  to the trainer's train_args. Two stages, two schemas.

## Original unit 2: a generated dataset is ready only when every published row is valid

This is the durable topic-level contract for the dataset-readiness half of
the first red-team queue. The state ledger recorded the mechanism, but this
file previously described the fail file without specifying how it controls
publication and training.

The current failure path is reachable and silent. A provider failure marks
the corresponding fail-file entry and leaves a zero-filled data-vector row.
The generator then calls `MPI.Finalize()` and exits with status zero
unconditionally. `emulator/data_staging.py` does not read the fail file, so
the trainer can treat a fabricated zero row as ordinary science. A separate
input error has the same trust shape: a `--boundary` value `<= 0` or `> 1` is
silently replaced by `1` instead of being rejected (`1` also takes the
fallback branch but is the valid unchanged endpoint). At the audited HEAD,
`generator_core.py` still contains both the unconditional `exit(0)` tail and
the boundary rewrite, and the staging module still contains no fail-file
consumer.

Required contract:

1. A row is successful only if the provider lifecycle accepted that exact
   cosmology and the complete stored payload passes the family's publication
   predicate after its storage-dtype cast. A returned array or a finite
   subset is not enough.
2. Failure state carries the original row identity and is part of the same
   checkpoint/dataset generation as the parameter table, axes, and payloads.
   Missing, stale, wrong-length, or forged failure metadata makes the set
   unreadable.
3. A production-ready marker is published only when the requested post-cut
   count is made entirely of successful rows. A run that ends incomplete
   exits nonzero or publishes an explicit non-ready status. Checkpoint
   material may retain failed proposals for an explicit retry, but it is not
   training input.
4. Every loader and staging entry loudly excludes **or** rejects flagged rows
   before `n_train`/`n_val`, sampling, thinning, centering, or device work. If
   exclusion is the chosen policy, the successful pool must still satisfy the
   requested count; a smaller dataset cannot be published silently. This is
   the original either/or ruling, not a later refusal-only redesign.
5. `boundary` is a native finite non-boolean real with `0 < boundary <= 1`.
   Validation happens before any output path is opened; no invalid value is
   rewritten to a default.
6. Refusal preserves every previously valid file byte-for-byte. This unit
   composes with checkpoint-set transactionality and row authenticity; neither
   substitutes for the readiness verdict.

This unit consumes the existing owners rather than cloning them: unit 33
(45M-64/70) owns the provider-lifecycle verdict; unit 56 (45M-48) owns the
post-cast payload predicate; the 20M-15 amendment owns checkpoint-ingress
revalidation; unit 82 owns row authenticity; and the amended 45M-81 contract
owns RNG continuation. Dataset readiness combines those exact verdicts into
the final ready/non-ready state. An "almost equivalent" local predicate would
recreate the drift this program is trying to remove.

Required gates use a small deterministic dataset: one provider rejection that
leaves a zero payload; a nonzero fail flag paired with an otherwise plausible
row; an all-success forged fail file paired with an invalid payload; missing
and wrong-length failure metadata; and a fully successful control. Each
failure must produce a non-success readiness verdict and either a loader
refusal naming the original row or a loudly reported exclusion that still
delivers the exact requested successful-row count. Missing or wrong-length
metadata errors name the path and expected/observed structure rather than
inventing a row number. Boundary controls cover zero, a negative value, a
value above one, NaN, a boolean, the valid endpoint `1`, and a valid interior
value. A mutation that restores unconditional exit zero or removes the
staging-side readiness check must turn the board leg red.

The MPS `sigma8` half that originally shared unit 2 has its complete modern
contract in `families-background-mps.md` (including the later 45M-67 domain
extension). Keeping the two halves in their scientific owner notes avoids
making generator readiness depend on one derived MPS quantity.

## Staging (emulator/data_staging.py)

- Memmap the dv dump (never loaded whole); params load to RAM;
  phys_cut_idx is params-only. `stage_source(C, dv, idx, ram_frac)`:
  if the sorted-unique row subset fits ram_frac of free RAM,
  materialize the COMPACT copy and reindex locally; else keep the
  memmap with global idx — invisible to consumers because everything
  touches C/dv THROUGH idx. The source dict carries the centers
  (C_mean, dv_mean via chunked stream_stats).
- Multi-GPU rule: parallel paths set ram_frac = 0 — np.asarray on a
  shared memmap makes a PRIVATE per-worker copy (the shared-budget
  trap in parallel form).
- `load_source` returns `dump_rows` (sorted-unique disk rows) so a
  sibling dump file — the MPS syren base dumps — stays row-aligned
  through a shuffled staging.
- The .paramnames cross-check: when the sidecar exists, its first
  column must match the covmat-header names ORDER INCLUDED; mismatch
  is a loud error naming both lists.

## Red-team staging gaps (verified 2026-07-12, open)

### A checkpoint is not yet one committed dataset

The checkpoint census and publication order do not describe one atomic
generation of the dataset:

- `__load_chk` requires the parameter chain, ranges, covmat, fail file,
  and the family's data-vector members, but omits `.paramnames` and the
  background/MPS grid sidecars. A resumed MPS or background job can thus
  accept a checkpoint whose axis files are missing or stale; the trainer
  later reads those files as the column coordinates.
- Multi-member families replace/flush each `.npy` independently. Append
  first extends the parameter text, then the fail file, then each data
  member in sequence. A process death can expose a mixture of old and new
  generations; row counts catch some interruptions but same-shaped stale
  members and old sidecars remain admissible.
- Most seriously, `__run_mcmc` catches ANY checkpoint-load exception,
  prints it, sets `loadedfromchk = False`, and continues down the fresh-run
  path using the SAME output roots. With `--loadchk 1 --append 1`, a
  partial append or missing member can therefore turn an intended resume
  into a from-scratch replacement of the old chain and dump set.

Required contract: a manifest binds every member (params, paramnames,
ranges, covmat, fail flags, all quantity arrays, every axis sidecar) to a
shared dataset id, config/order digest, dimensions, and per-file digest.
Publish a new manifest last after all temporary members are durable; load
only the manifest's exact generation. A requested load/append that cannot
validate must stop nonzero without modifying the prior files. Gates inject
failure after each append/publication boundary, remove or swap an axis
sidecar, permute same-width parameter order, and prove both loud refusal
and survival of the previous good generation.

### The ordinary `.1.txt` naming check is bypassed

The generator writes `X.1.txt` and `X.paramnames`. Ordinary
`load_source` looks only for `X.1.paramnames`, so the advertised
sidecar-vs-covmat order check is skipped on the standard generated
file name. `load_scalar_source` already contains the correct resolver:
try the exact stem, then strip a purely numeric chain suffix and try
the chain root. One shared resolver must serve both loaders. The gate
uses `X.1.txt` + `X.paramnames` with a deliberately permuted order and
requires `load_source` to reject it. Generated training data should
require the sidecar; any legacy no-sidecar escape must be explicit and
loud in the banner, never inferred from absence.

### Grid2d staging defeats its own memory ladder

The documented production MPS grid is 122 redshifts x 2,000 k values;
the shipped trainer asks for 50,000 rows and `k_stride: 10`. The raw
selected float32 surface is 45.449 GiB. `_grid2d_law_rows` currently:

1. calls ordinary `load_source(..., with_means=True)`, which streams
   statistics over every unthinned raw column even though that mean is
   discarded;
2. fancy-indexes every selected raw row and casts the whole selection
   to float64 (90.897 GiB);
3. fancy-indexes the whole base selection and casts that to float64
   too (another 90.897 GiB, simultaneously resident);
4. forms the ratio/log, and only then keeps the 201 k columns selected
   by the stride (the final float32 target is about 4.568 GiB);
5. forcibly replaces the source with an in-RAM array, ignoring the
   `ram_frac` / memmap decision the preceding staging step made; and
6. later passes the whole thinned matrix through
   `Grid2DGeometry.from_targets`, which casts it to float64 again to
   compute center/scale instead of using streamed statistics.

This is a production blocker and contradicts every "never loads the
dump whole" statement. The required behavior is outcome-based:

- derive and validate the kept `(z, k)` columns first; read only those
  columns for bounded row chunks from both raw and base memmaps;
- perform positivity checks and the law transform on consumed entries
  in those chunks; never materialize an unthinned selected matrix;
- compute `dv_mean` from the thinned law-space rows, not from the raw
  dump, compute scale/constant pins with the same bounded statistics,
  and preserve exact `C` / `dv` / `dump_rows` alignment;
- honor `ram_frac` after transformation: the final thinned source may
  be resident or memmapped, and train/validation workers must not each
  create an unconditional private full copy;
- validate both row count and width for every raw/base/grid member.

Add this as a leg of `mps-identity`, not a new board item: a synthetic
122 x 2,000 grid with stride 10, a deliberately tiny memory budget, and
guarded memmap reads must prove that every read is row-chunked and
column-thinned, the result has 122 x 201 columns, its values/mean equal
a small direct known-answer calculation, and the low-RAM result remains
disk-backed. The
test must fail on the current whole-selection implementation.

This unit closes ordinary grid2d training, fine-tuning, and frozen-base
transfer only. The optional grid2d PCE fit has a separate scale blocker:
`_fit_diag_pce` materializes every thinned target, moves it all to the
GPU, copies it back as float64, and runs a dense SVD. At the documented
production shape the target alone is about 4.568 GiB float32 / 9.135 GiB
float64, before SVD workspace, and cannot fit the 12 GiB test cards.
The identity gate proves only smoke-scale algebra. Do not claim
production-sized MPS PCE until a separate streamed/randomized low-rank
fit contract and accuracy calculation are designed; do not smuggle that
research change into the bounded-staging unit.

#### Bounded staging resume (2026-07-12, Opus) — awaiting Architect audit

Built. Five surfaces:

- `emulator/experiment.py` `_grid2d_law_rows` rewritten: it builds the
  kept `(z, k)` columns FIRST (k_stride, top edge always kept), then
  reads only those columns in bounded row chunks (`_GRID2D_CHUNK_BYTES`,
  256 MiB of float64 per read -> derived chunk height) from the raw and
  base memmaps as a single `raw[rows[:,None], cols[None,:]]` gather (never
  a whole row block), takes `log(raw / base)` per chunk (raw itself under
  law "none"), checks positivity per chunk with the original error text,
  and writes the thinned float32 rows into a result that is resident if
  it fits `ram_frac` else a disk-backed temp memmap (unlinked at exit).
  The per-point law-space moments (mean + population std) are STREAMED
  over the same chunks. New signature adds `with_means` (train streams
  the moments + `dv_mean`; val only forms the rows). Handles both a
  memmap source (read by `dump_rows`) and a RAM-staged compact source
  (read by local arange).
- `emulator/data_staging.py` `load_source` gains `stage_dv=True`; the
  grid2d path (`stage_train` / `stage_val`) passes `stage_dv=not
  self._grid2d`, keeping the raw dump a memmap (no unthinned 45 GiB
  float32 selection) and skipping the discarded raw `dv_mean`.
- `emulator/geometries/grid2d.py`: `from_stats(center, scale, ...)` added
  as the SINGLE home of the constant-pin / dead-dump rules; `from_targets`
  now computes the two moments and delegates to it. `from_config`'s plain
  grid2d branch builds the geometry from the streamed
  `self._grid2d_center` / `_scale` via `from_stats`, so the thinned
  surface is never read whole (the disk-backed result is never
  materialized). Finetune / transfer branches unchanged (they pin the
  source geometry).
- `gates/checks/mps_identity.py`: `check_bounded_staging` — 122 x 2,000,
  stride 10, tiny budget, a `_GuardProxy` that fails a whole-block /
  over-tall / unthinned read. Asserts the 122 x 201 result, values +
  streamed mean vs an independent known answer, disk-backed low-RAM
  result, and that the guard trips on the old `mm[rows]` pattern (so the
  leg fails against the pre-fix code). Part two runs the real
  `load_source(stage_dv=False)` memmap branch (resident vs disk by
  `ram_frac`). Existing `check_staging` updated for the new signature.
- `gates/board.py`: the cmb-identity maps rider (eq-6 now five legs).

Scope note for the audit: the boundary excludes the "geometry" queue
unit, which is item 11 in state-2026-07-11-and-next.md (covariance SPD /
block-whitening / rebuild validation). `from_stats` is not that unit —
it is this unit's clause 3 ("constant pins computed streaming, never via
a materialized unthinned selection") and the note's "compute scale /
constant pins with the same bounded statistics". Without it the
disk-backed result would be re-materialized at geometry build, defeating
the ram_frac honoring. Flagged for the Architect to confirm the read.

Known limitation (flagged, not fixed): the disk-backed temp memmap is
unlinked at process exit; a learning-curve sweep that re-stages many
times ON A BOX TOO SMALL to hold the 4.568 GiB thinned result would
accumulate temp files until exit. That path is the tiny-RAM edge (any
real MPS training box holds the thinned result resident); a per-restage
unlink is a small follow-on if it ever matters.

Mac gate (numpy exec-probes of the shipped bodies, no torch):
`probe_grid2d.py` 10/10 PASS (bounded transform bitwise vs the known
answer, reads chunked + thinned, disk-backed under tiny budget /
resident under ample, streamed moments = center + population std of the
answer, law-none passthrough, guard trips on the old read);
`probe_grid2d_geom.py` 6/6 PASS (from_stats pin / dead-dump / length
rules, from_targets delegates the two moments). `py_compile` +
`compileall` clean on all five files.

Close (user-run, workstation): `python gates/run_board.py --force-rerun
mps-identity` — the bounded-staging legs ride mps-identity; return the
raw log, close requires green. (transfer-identity remains the standing
open red from the prior unit, unrelated.)

### Bounded-staging Architect audit (2026-07-12, Fable): STRUCTURE ACCEPTED, REVISION REQUIRED before landing

Audited against the raw diff (eight files; the handoff said six — the
transfer-fixture fix and its resume rode along, see below). Verdict:

- Clauses 1, 2, 4, 5 VERIFIED in the diff: kept columns built from the
  sidecars + k_stride before any read; every read
  `raw[rows[:, None], cols[None, :]]`; chunk height from
  _GRID2D_CHUNK_BYTES / thinned width; stage_dv=False keeps the raw
  dump a memmap; per-chunk width/rows/positivity with the original
  error texts; tempfile memmap + atexit unlink. The gate leg's
  _GuardProxy + the must-fail-on-old sub-check match the acceptance
  contract.
- **Clause 3 is the blocker (the red team's in-flight audit,
  Architect-CONFIRMED by independent reproduction):** the streamed
  moments use the naive one-pass `(s2 - s1*s1/n)/n` with a
  clamp-to-zero. On a high-offset small-spread column (float32 1e8
  alternating with 1e8+8, true population std exactly 4.0) the naive
  form returns 3.9659 in the Architect's single-sum probe and 4.1279
  in the red team's chunked probe — the answer DEPENDS ON SUMMATION
  ORDER, proving non-equivalence to the former Y.std(0) contract, and
  under other orderings goes negative so the clamp converts a
  genuinely varying column into a FALSE constant pin (worst under law
  none, where physical spectra carry large offsets). REVISION: replace
  s1/s2 with per-chunk mean/M2 merged pairwise (Chan/Welford),
  float64 accumulation; reproduce float64 np.std(ddof=0) within a
  stated tight tolerance; the clamp may cover only the bounded
  round-off residue of the STABLE accumulator. Red legs (must FAIL
  the s1/s2 form): the 50k-row 1-ULP fixture; uneven chunk sizes and
  orderings; a constant column; the ordinary log-ratio fixture;
  streamed center/scale + the resulting encode vs the materialized
  known answer.
- Deviation 1 (Grid2DGeometry.from_stats + the from_targets
  delegation): SCOPE READ CONFIRMED — the excluded geometry unit is
  the covariance-SPD / block-whitening / from_state-validation item;
  the pin/scale construction is this unit's clause 3. Single-sourcing
  the pin and dead-dump rules through from_stats is accepted.
- Deviation 2 (temp files accumulate across re-stagings on a
  too-small box until process exit): accepted as flagged; the
  per-restage unlink is a recorded follow-on, not a blocker.
- The folded-in transfer-fixture fix (gates/checks/transfer_identity.py):
  matches the Architect's spec verbatim (a plain grid base saved
  without transfer_base; the chaining refusal keeps its own leg) —
  ACCEPTED as-is; it lands with this unit's revision in one commit.
- Prose rider joining the revision (the red team's terminology
  handoff): every HUMAN-FACING "oracle" in gates/board.py (maps,
  docstrings, log labels — including the five-leg maps line this unit
  edits) and gates/checks/cmb_identity.py (module docstring, comments,
  report text) becomes "independent known-answer calculation/check";
  callable identifiers (check_covariance_oracle, _oracle_truth) stay.
- Close after revision: ONE commit (staging revision + fixture fix +
  prose rider), then the user runs
  `python gates/run_board.py --force-rerun mps-identity cmb-identity transfer-identity`
  — three raw logs; 32/32 expected.

### Red-team amendment to bounded grid2d staging: stable moments over the stored payload

The Implementer's in-flight bounded-read design (kept columns chosen
before every read; row chunks; RAM-or-memmap result) closes the memory
mechanism, but the reviewed draft computes population variance as
`(s2 - s1*s1/n)/n` and clamps negatives to zero. That subtraction is
not equivalent to the former `Y.std(0)` path: for 50,000 alternating
float32 values 1e8 and its next representable value, the true std is
4.0 while the draft returns 4.127875...; at one million rows it becomes
negative and the clamp turns a varying column into an exact constant.
That can change whitening and the physical-constant/dead-dump decision.

The draft also accumulates moments from the pre-cast float64 law chunk,
then trains on its float32 cast. The old path cast the law rows to
float32 first and computed geometry statistics from those exact stored
targets promoted to float64. Required before landing: merge chunk
mean/M2 with a stable Chan/Welford algorithm in float64, but feed it the
exact float32 payload written to the result. Gates compare center,
population std, constant mask, and encoded rows against the former
materialized-float32 path across uneven chunks, a high-offset/one-ULP
spread, a true constant, and ordinary log-ratio rows. An analytic
pre-cast log result alone is not the reference.

**Architect adjudication (Fable, at the merge):** ADOPTED WHOLE into
the revision above. The cancellation half was independently confirmed
before this recording arrived (the audit verdict's probe); the
FLOAT32-PAYLOAD half is NEW and accepted — the in-flight draft indeed
accumulates from the float64 pre-cast chunk while storing (and later
training on) its float32 cast, so the revision's Chan/Welford
accumulator must be fed `law_chunk.astype("float32")` promoted back to
float64 (the exact stored payload), and the gate's reference is the
former materialized-float32 path, never an analytic pre-cast result.
This clause is binding on the revision handoff.

#### Revision resume (2026-07-12, Opus) — awaiting Architect re-audit

Revised. `_grid2d_law_rows` now streams the geometry moments with a
module-level `_merge_chunk_moments` (Chan's parallel form of Welford, all
float64): per chunk it computes the block mean and the block M2 about
that mean, then merges into the running `(count, mean, M2)` — no
`sum(x^2)` to cancel against `sum(x)^2`. The final population variance is
`M2 / count`, and the clamp is `np.maximum(var, 0.0)`, which now only
absorbs the round-off residue of an exactly constant column (the stable
M2 is accurately positive for a varying one). Per the binding
float32-payload clause, the merge is fed the EXACT stored payload: after
`law_rows[a:b] = law_chunk.astype("float32")` it reads `stored =
np.asarray(law_rows[a:b], dtype="float64")` and merges THAT, so the
moments match the former materialized-float32 `from_targets` path (which
promoted the stored float32 targets), never the pre-cast float64 chunk.

Gate: `gates/checks/mps_identity.py` gains `check_stable_moments` (+ the
`_run_law_none` helper) — the red legs that fail the s1/s2 form and pass
this one: (1) the 50,000-row float32 1e8/1-ULP column across several
chunk heights, streamed scale = np.std(ddof 0) = 4.0 with no false pin,
the chunkings agreeing; (2) a constant column pins while a varying
large-mean column does not (through `from_stats`); (3) the ordinary
log-ratio fixture, streamed scale = np.std and the `from_stats` encode
matching the materialized standardization. Prose rider (this unit's half,
board.py's was already landed in the fixture commit): every human-facing
"oracle" in `gates/checks/cmb_identity.py` (module docstring, comments,
report line) is now "known-answer" / "these legs"; `_oracle_truth` and
`check_covariance_oracle` stay.

State note: the transfer-fixture fix + board.py five-leg cmb metadata
already landed as their own commit (6e9757f, Architect-audited), so this
held commit is the whole bounded-staging unit (revised with the Chan
moments), the cmb_identity.py "oracle" prose rider, and the mps-identity
board registry prose for the new legs (gate_mps_a docstring + the
mps-identity maps field) — seven files (experiment / data_staging /
grid2d / mps_identity / cmb_identity / board / this note).

Corrected queue order (Architect, at this revision): after the
bounded-staging revision lands, the next units are finite training/
evaluation contract (CRITICAL) -> selection-record truth (CRITICAL,
depends on the finite contract) -> covariance-input validation. The
training-truth pair protects every production run and precedes the
covariance-input unit (my earlier order had them reversed).

Mac gate (numpy exec-probes of the shipped bodies, no torch):
`probe_grid2d.py` 10/10 and `probe_grid2d_geom.py` 6/6 still green;
`probe_grid2d_moments.py` 8/8 PASS — 1e8/1-ULP stable std exactly 4.0
(the old s1/s2 form gives 3.9659, matching the audit probe), uneven
chunkings agree, constant -> 0 and varying-large-mean -> nonzero, and the
real `_grid2d_law_rows` streamed scale + encode match the materialized
answer bitwise. `probe_cm11a.py` 5/5 still green after the prose rider.
`py_compile` clean on all five files.

Close (user-run, workstation, one commit): `python gates/run_board.py
--force-rerun mps-identity cmb-identity transfer-identity` — three raw
logs; 32/32 expected.

#### Revision re-audit (2026-07-12, Fable): NUMERICS ACCEPTED; TWO AMENDMENTS BEFORE COMMIT

Independent probe (`audit_grid2d_revision.py`, exec-extraction of the
shipped bodies, stub psutil — not a new dependency, data_staging.py
already imports it): 23 of 24 legs pass, and the one failure is the
deliberate reproduction of a gate-fixture defect, not a code defect.

What the probe proved about the shipped code, all independently
re-derived:

- `_merge_chunk_moments` reproduces float64 `np.std(ddof 0)` of the
  stored payload on the 1e8/1-ULP fixture at rtol 1e-9 across chunk
  heights 7 / 337 / 4096 / whole, the chunkings mutually identical at
  rtol 1e-12. The old s1/s2 form on the SAME fixture returns column-0
  std 0.0 / 4.5795 / 13.7384 at those chunkings against true 4.0 — the
  0.0 is the false constant pin reproduced verbatim, and the spread is
  the order-dependence. Chan's M2 is non-negative by construction, so
  the `np.maximum(var, 0.0)` clamp is provably a no-op — strictly
  stronger than the contract asked.
- The float32-payload clause holds and is DISCRIMINATED: on a fixture
  whose stored-float32 std differs from the pre-cast float64 std by
  12% (law values 100 + 1e-5 noise against ULP(100) = 7.6e-6), the
  shipped scale matches the stored payload and fails against the
  pre-cast reference.
- `_grid2d_law_rows` end to end: values bitwise against the direct
  known answer on both the memmap and RAM-staged-compact branches
  across three chunkings; C compacted in dump_rows order with local
  arange idx; dv_mean = float32(mean of stored payload); kept axes
  stashed; tiny-ram_frac result a disk-backed memmap with bitwise
  values; with_means False leaves the moments None; the positivity and
  base-too-short guards raise.
- `from_stats` / `from_targets` single-sourcing: identical center /
  scale / mask through both routes; wholly-constant raises; length
  mismatch raises.

Amendment 1 (gate fixture, REQUIRED): `check_stable_moments` leg 2
CRASHES the whole mps-identity gate on the workstation. The shipped pin
threshold is RELATIVE — `tiny = 8 * eps32 * |center|`, which is ~95.4
at center 1e8 — so the leg's 1-ULP column (streamed scale 4.0) is
classified constant, and with the other column exactly constant,
`zero.size == n_out` fires the dead-dump ValueError (reproduced in the
probe with the leg's exact numbers). Note the Mac probe was green
because it checked the STREAMED scale (nonzero, correct); the gate leg
goes through `from_stats`, where the relative threshold rules. The
pinning of a 1-ULP spread is CORRECT behavior — a column varying by one
float32 ULP is numerically constant for standardization — so the fix is
the fixture, not the rule: the varying column's spread must sit clearly
above the relative threshold (e.g. alternate +-1024 at 1e8: std 1024 ~
10.7x tiny). Binding requirements: (i) no dead-dump crash; (ii) an
above-threshold varying large-mean column asserted NOT pinned; (iii)
the relative-threshold rule documented in the leg — recommended shape
is three columns [constant -> pins, 1-ULP at 1e8 -> pins BY THE
RELATIVE RULE (assert it, with a comment saying why that is correct),
+-1024 at 1e8 -> must not pin], which turns the discovered behavior
into a documented leg and keeps the whole-surface guard out of reach.

Amendment 2 (docstring, REQUIRED): the `_grid2d_law_rows` shape-flow
diagram still says "accumulate s1 / s2 (streamed mean / std)" and its
legend still defines "s1 / s2 = running sum and sum-of-squares"
(experiment.py:2888 and :2895 at this diff) — the accumulator the
revision deleted. Update the diagram line and the legend to the Chan
merge's (count, mean, M2); every symbol in the legend, per the house
shape-flow rule.

Everything else stands as delivered: the board registry prose
(gate_mps_a docstring + maps), the cmb_identity.py prose rider (my full
uncut sweep of `-i "oracle"` over gates/ + emulator/ finds only the
identifiers `_oracle_truth` / `check_covariance_oracle` / the importlib
tag `cmb_cov_oracle`, and board.py:1012 naming the function — exactly
the rider's carve-out), and the corrected queue order. The commit stays
HELD for the two amendments; on their delivery I re-check just those
two spots and commit the whole seven-file unit myself.

**Amendments applied (2026-07-12, Opus) — awaiting the two-spot re-check.**
Amendment 1: `check_stable_moments` leg 2 is now a THREE-column fixture
that keeps `zero.size < n_out` (no dead-dump crash) and documents the
relative rule — col 0 exactly constant pins; col 1 a 1-ULP spread at 1e8
(streamed std 4, ~4e-8 relative, below float32 precision) pins BY THE
RELATIVE RULE, asserted with a comment on why that is correct
standardization; col 2 a std-1024 spread at 1e8 (~10.7x tiny) is
resolvable and must NOT pin. The leg asserts `const_mask == [True, True,
False]`. Amendment 2: the `_grid2d_law_rows` shape-flow diagram + legend
now read "merge (count, mean, M2) over the STORED float32 rows" and
define count / mean / M2 as the Chan/Welford aggregate (the deleted
s1 / s2 is gone from both the arrow line and the legend). Nothing else in
the seven files touched. Mac re-verification: `probe_grid2d_geom.py` now
9/9 — it reproduces the two-all-sub-threshold-column whole-surface raise
(the pre-fix leg-2 crash) and proves the three-column
`const_mask [True, True, False]`; `probe_grid2d.py` 10/10,
`probe_grid2d_moments.py` 8/8, `probe_cm11a.py` 5/5 unchanged;
`py_compile` clean on all seven files.

**Amendments re-checked (2026-07-12, Fable): ACCEPTED, unit committed.**
Two-spot re-check done with independent probes of the shipped bodies:
the three-column fixture returns `const_mask [True, True, False]`
through the real `from_stats` (pinned scales set to 1.0, column 2
keeping 1024.0), and the full 23-leg audit probe stays green on the
current `_grid2d_law_rows` / `_merge_chunk_moments` bodies, proving the
amendment touched nothing else. The shape-flow diagram + legend now name
the (count, mean, M2) Chan aggregate; `grep s1` over experiment.py = 0.
The other five files' diffs are byte-identical to the accepted delivery
(stat-verified). One collateral fix at this re-check: the append above
had eaten the "File names and row counts" section header below —
restored. Close (user-run, workstation): `python gates/run_board.py
--force-rerun mps-identity cmb-identity transfer-identity`; 32/32
expected.

#### Close REOPENED (2026-07-12, red team; Architect-ADJUDICATED): disk-backed staging files accumulate across sweep points

Red-team reopen of the c03a084 close, verified whole and ACCEPTED. The
numerical contract is untouched; this is lifecycle fallout of the new
disk-backed result. The Architect's original deviation-2 acceptance
("temp files accumulate for the process") is hereby CORRECTED: it was
scoped to a single train run (at most one train + one val file per
process) and is falsified in the sweep context — the shared N-train
sweep reuses ONE experiment per lane, calls `stage_train(n_train=N)`
per point, and cleans up with `exp.train_set = None`
(cosmic_shear_sweep_ntrain_emulator.py:160 and :168, all thin family
sweeps import this main), so every low-RAM point orphans its
`.g2law.dat` until worker exit. At the production 50,000 x 24,522
shape each file is 4,904,400,000 bytes (~4.57 GiB); a lane sweeping
several sizes can exhaust temporary storage with RAM fully bounded.
The lane's `except Exception` keeps workers alive across failures, so
failed points accumulate files too. Verified mechanics: mkstemp +
atexit-only at experiment.py:2977-2981; dropping the memmap reference
closes the mapping but never unlinks; the gate's disk-backed leg
asserts only `isinstance(np.memmap)` — no restage or absence
assertion, so the leak stays green.

Micro-revision spec (Architect; land BEFORE the three-gate rerun so
the workstation run is spent once):

- Ownership on the EXPERIMENT, not the source dict: two slots,
  `self._grid2d_train_tmp` / `self._grid2d_val_tmp` (path or None) —
  train and val lifetimes stay independent (the sweep lane keeps its
  val staging across points). `_grid2d_law_rows` reports the created
  temp path to its caller; stage_train / stage_val own the slots.
- Supersede-on-restage: stage_train releases the train slot BEFORE
  staging a replacement (close the mapping where we still hold it,
  unlink the path; `_unlink_quietly` tolerates the already-gone);
  stage_val likewise for the val slot. The current mapping is never
  released while its point's loaders run — release happens only at
  the NEXT staging call or an explicit release.
- Public release for the drivers: `release_train_staging()` /
  `release_val_staging()`; the shared sweep lane's cleanup block calls
  `release_train_staging()` where it now sets `exp.train_set = None`
  (both the success and the except paths reach it), so a failed
  training releases that point's file and a finished lane holds at
  most its val staging.
- Failure hygiene inside the transform: wrap the chunk loop so any
  exception after the temp file exists unlinks the partial file
  before re-raising (transform, positivity, base-shape errors).
  atexit stays registered as FALLBACK ONLY (it already tolerates
  double-unlink); SIGKILL/OOM/MPI.Abort remain out of scope, as
  before.
- Red legs (adopted verbatim): (a) low-RAM staging twice on one
  experiment — the first path is absent, the second readable;
  (b) three N-train sweep points — live temp count and bytes bounded
  to current sources, never cumulative; (c) injected mid-transform
  failure leaves no file; (d) failed training after successful
  staging releases that point's train file through the lane-style
  cleanup; (e) resident-RAM control creates no temp file and stays
  byte-identical. Then rerun the existing bounded-read,
  stable-moments, cmb-identity, and transfer-identity legs.

Amended close (user-run, workstation, after THIS lands): the same
`python gates/run_board.py --force-rerun mps-identity cmb-identity
transfer-identity`; 32/32 expected. Do not spend the rerun before the
micro-revision.

##### Lifecycle micro-revision resume (2026-07-12, Opus) — awaiting Architect audit

Built; numerics frozen (`_merge_chunk_moments`, the chunk read/merge
order, and the geometry path untouched — the 4 numeric Mac probes rerun
green). Files:

- `emulator/experiment.py`: two ownership slots in `__init__`
  (`self._grid2d_train_tmp` / `self._grid2d_val_tmp`, path or None).
  `_grid2d_law_rows` now RETURNS the created temp path (None when
  resident) and wraps the chunk-loop transform in `try/except
  BaseException` that drops the mapping and `_unlink_quietly`s the
  partial file before re-raising (failure hygiene; atexit stays as
  last-resort fallback only). `stage_train` / `stage_val` call
  `release_train_staging` / `release_val_staging` (supersede) before
  staging and store the returned path in their own slot. New public
  `release_train_staging()` / `release_val_staging()` — path-only unlink,
  clear the slot, idempotent; independent train / val lifetimes (POSIX
  unlink-while-mapped frees space on close, the name goes immediately).
- `cosmic_shear_sweep_ntrain_emulator.py`: the per-point cleanup calls
  `exp.release_train_staging()` where it drops `exp.train_set` (reached
  on both the success and the `except` paths); val staging is kept across
  points, so it is NOT released there.
- `gates/checks/mps_identity.py`: `check_staging_lifecycle` — the five
  red legs on the REAL stage_train / release paths: (a) double low-RAM
  staging (first file absent, second readable); (b) three sweep points,
  live temp count/bytes bounded to one point, 0 at end; (c) a
  mid-transform positivity failure leaves no temp file; (d) failed-point
  lane cleanup releases the train file; (e) resident control makes no
  temp file and is byte-identical. Wired into main(); module docstring
  bullet added.

Design note (resolved while coding): `release_*_staging` only unlinks the
tracked PATH and never touches `train_set["dv"]` — mid-`stage_train`,
`train_set["dv"]` is the fresh RAW-dump memmap, not the old temp file, so
dropping it would corrupt the in-progress transform. The mapping is
closed by `train_set` reassignment / None-ing (sweep) before release.

Mac gate: `probe_grid2d_lifecycle.py` 5/5 (disk-backed returns a live
path / resident returns None; mid-transform failure unlinks the partial
with 0 orphans; release unlinks + clears + idempotent; train/val slots
independent). The stage_train supersede wiring and the sweep-lane cleanup
are torch (load_source) — the gate's `check_staging_lifecycle` runs them
for real on the workstation. `probe_grid2d` / `_moments` / `_geom` /
`probe_cm11a` rerun green (numerics unchanged); `py_compile` clean on
experiment.py, the sweep driver, and mps_identity.py. Held for audit.

Note: the finite training/evaluation contract (unit 14) is HELD in the
same worktree on separate files (training.py + training-stack.md) — no
overlap with this micro-revision's files.

##### Audit (2026-07-12, Fable): ACCEPTED — unit committed; the amended close is spendable

- Independent probe (exec-extracted shipped bodies, stubbed psutil, no
  torch): all sub-checks green over five legs — resident returns None
  with bitwise passthrough and np.mean / np.std(ddof 0) moments;
  disk-backed returns a live path whose contents are bitwise the
  resident answer; the real release unlinks + clears + is idempotent; a
  disk-backed positivity failure propagates ValueError with zero
  orphans; a resident failure makes no file; train / val slots
  independent through the real release methods.
- Wiring read-verified: stage_train / stage_val supersede-on-restage
  (release before staging, slot assigned from the return); the sweep
  lane's release sits AFTER the try/except
  (cosmic_shear_sweep_ntrain_emulator.py:174) so success and failed
  points both reach it; a failed stage_train leaves the slot None, so
  the lane release is a no-op, never a double unlink.
- Gate crash-risk check (the amendment-1 class from the parent unit):
  an AST fan-out over stage_train -> _grid2d_law_rows ->
  release_train_staging shows the only self attributes read beyond the
  gate fake's set are `outputs` (scalar branch only; the fake sets
  _scalar False) and `train_set` (stored before every read) — so
  check_staging_lifecycle cannot AttributeError on the workstation. The
  fixture's params layout matches the file's established IN_NAMES
  pattern, and train_set["dump_rows"] survives the transform (read-only
  there), so leg (e)'s known answer is well-formed.
- Numerics frozen confirmed: the chunk-loop diff is indentation-only
  (the try wrapper plus the return); merge order and the geometry path
  untouched.
- Design-note ruling: the path-only unlink (release never touches
  train_set["dv"]) is ACCEPTED with the Implementer's reasoning —
  recorded as the unit's semantics, not a deviation.
- One rider, NOT a hold: gates/board.py:1427-1431 (the mps-identity
  `maps` string) still names only the bounded-staging + stable-moments
  legs; add the staging-lifecycle leg to that string on the next
  Implementer commit touching gates/ (docs parity, the parent unit's
  own convention).
- The mkstemp -> np.memmap window (an allocation failure between file
  creation and the try wrapper) is covered by the atexit fallback only;
  reviewed and accepted — the realistic failure (positivity, mid-loop)
  is inside the wrapper and gate-checked.

The amended close is now SPENDABLE (user-run, workstation):
`python gates/run_board.py --force-rerun mps-identity cmb-identity
transfer-identity`; 32/32 expected.

### File names and row counts do not prove dataset identity

`load_source` checks only that parameter and dv row counts match. A
parameter table from run A and a same-shaped dv from run B silently
train as pairs; the MPS raw/base/grid/fail members have the same issue.
The finalized generator output needs one manifest identity covering the
parameter order, row count, parameter file, every dv/base/grid member,
and the failure mask, with strong content digests. Staging verifies the
bundle before cuts or shuffles. Because the dumps are large and sweeps
spawn many readers, verification must be performed once per immutable
file identity and shared/cached safely rather than re-hashing the full
bundle in every worker. The gate swaps same-shaped dv files between two
fixtures and requires a pre-staging identity failure.

### Family sidecar paths and validation axes are not bound (red-team continuation — ADJUDICATED: these are queue units 25 + 26 below)

**Architect adjudication (Fable, at the merge):** verified and
numbered before this recording merged — the same two findings are
queue units 25 ("Nested data paths never resolve") and 26
("Validation grid axes are never identified"), specs later in this
note, cluster-ruled with the checkpoint manifest (8) and generator
ingress (17) as one file-set authenticity boundary. This section
stands as the red-team record; the numbered sections are the specs of
record.

`resolve_cocoa_config` resolves the five flat dv/parameter/covmat paths
under `<project>/chains`, but leaves `data.cmb.covariance`,
`data.grid.z_file`, and grid2d's `z_file` / `k_file` / `train_base` /
`val_base` cwd-relative. The shipped examples use bare names and tell
the user to launch from `$ROOTDIR`; a direct resolver probe therefore
made `train_dv` absolute while leaving `z_file` and `train_base` bare.
Existing gates hide the split by constructing absolute nested paths.
One dotted-path registry must resolve every file-valued leaf against its
documented base (absolute paths unchanged), and missing-file errors name
the dotted key.

There is also only one background z sidecar and one grid2d (z,k) pair:
the training axes are reused to interpret validation dumps. A validation
dump/base from a different generator run with the same width passes and
is scored on the wrong coordinates; CMB has the analogous same-width
ell identity gap. The committed dataset manifest should bind each
train/val raw/base dump to its own axis bytes, parameter order, failure
mask, settings, and generation id, then require exact train/val axis
equality before staging. Until that manifest lands, explicit validation
sidecars plus `array_equal` guards are the minimum. Red legs use shifted,
reversed, and permuted same-width axes and a swapped same-shaped val base;
byte-identical separately written axes pass.

### No-cut learning-curve pool counting is broken

`EmulatorExperiment.pool_size` correctly recognizes that scalar, CMB,
grid, and grid2d configs may omit `data.param_cuts`, assigns `{}`, and
then immediately reads `pc["omegabh2_hi"]`. Their family
`*_sweep_ntrain_emulator.py` wrappers call this before staging, so every
shipped no-cut example can die with `KeyError` before its first point.
The scalar branch also re-slices the chain positionally instead of
using the same sidecar-resolved input columns as `load_scalar_source`.

Pool counting must reuse the exact staging selection contract: no cuts
means the full row count; active cuts use the same named input columns,
formulas, and bounds as `stage_train`; scalar inputs come from the
sidecar resolver; and the reported pool must equal the maximum legal
`stage_train(n_train=...)`. Add one cheap no-cut and one active-cut pool
leg for each optional-cut family to its existing identity gate, plus a
thin-wrapper invocation that reaches `pool_size` without a GPU training
step. This is a small driver-truth unit, separate from bounded grid2d
staging.

## Nested data paths never resolve (red-team 2026-07-12, Architect-VERIFIED, open; the file-set authenticity cluster)

resolve_cocoa_config (emulator/cocoa.py ~135-139) rewrites ONLY the
flat _DATA_PATH_KEYS under data:; the nested file leaves —
data.cmb.covariance, data.grid.z_file, data.grid2d.{z_file, k_file,
train_base, val_base} — are never touched, and the family builders
np.load those strings directly. The shipped examples use bare
filenames + the documented run-from-$ROOTDIR workflow, so the
advertised CMB/background/MPS examples are internally split across
two path bases and fail unless launched from chains/ (the gates hide
it by writing absolute nested paths). The function's own docstring
claims it "rewrites every data-block file path" — false until this
lands.

Contract (the red-team block of record adopted whole): one dotted
path registry resolves EVERY file-valued schema leaf against the
correct project base; absolute paths pass through; errors name full
dotted paths; covariance + sidecars/base dumps live with the chain
products unless an explicit documented base says otherwise; persist
the resolved absolute consumed paths (or a clear portable-root form)
consistently. Red legs: each shipped CMB/background/MPS example under
a temp ROOTDIR with placeholder files in project/chains resolves
whole and loads from a different cwd; absolute nested paths pass
unchanged; missing files name the dotted key; a negative leg proves
the old cwd-relative location is not consulted. The resolver
docstring/README promise is corrected in the same unit.

## Validation grid axes are never identified (red-team 2026-07-12, Architect-VERIFIED, open; the file-set authenticity cluster)

Background has one z_file, grid2d one z_file/k_file — the TRAINING
axes interpret BOTH train and val dumps (the staging comment says it
outright: "val borrows the" training axes; experiment.py ~3145).
There are no validation-axis keys and no manifest digest, so a val
dump produced on different coordinates with the same column count
passes every width check and is silently scored on the wrong grid;
the val base can likewise come from a different same-shaped run; CMB
val rows are interpreted on the covariance ell grid with width-only
identity.

Contract (adopted whole): train and val dumps each carry/point to
their own axis identity, startup requires exact train/val equality
before staging. The PREFERRED home is the already-queued committed
dataset manifest (bind every raw/base dump to axis bytes, parameter
order, generation settings, fail mask; configs reference manifest
members); until it exists, explicit val sidecars + exact array_equal.
Red legs: same-width shifted/reversed/permuted val z; altered k;
swapped val base from another run; CMB same-width shifted ell; all
fail before geometry/training; byte-identical separately-written axes
pass; train/val raw/base/axis members share one manifest generation
id. SEQUENCING (red-team ruling, adopted): this unit + the checkpoint
manifest + generator ingress + the nested-path resolver are ONE
file-set authenticity boundary — the Implementer takes them as one
cluster.

## Tenth wave: validation leakage + data-control totality (red-team, Architect-VERIFIED; CRITICAL first clause; folded into the file-set authenticity cluster)

Four clauses, all confirmed, all joining the 8 + 17 + 25 + 26 cluster:

1. **Validation can BE the training set (CRITICAL).** stage_train and
   stage_val both recreate torch.Generator().manual_seed(the SAME
   split_seed) — experiment.py ~3086 / ~3142 / ~3169 — and the
   stage_val docstring states the unenforced premise verbatim ("the
   val file differs, so the same seed gives an independent
   selection"). No samefile/realpath/duplicate-payload check exists.
   Aliased or row-overlapping train/val paths make the same shuffled
   prefix the "validation" set: reported validation performance is
   then training performance, silently. Contract: reject train/val
   path aliases before staging; the manifest cluster binds stable
   row/sample identities and PROVES train/val disjointness (not just
   distinct filenames + matching axes); same-pool splitting is
   unsupported today and must be REFUSED (if ever supported, one
   explicit partition operation with a proven empty intersection).
   Red legs: identical paths; symlink/hardlink aliases; separately
   named duplicate payloads; partial row overlap; a valid disjoint
   pair.
2. **One-row contract contradiction.** validate_sizes permits
   n_train/n_val = 1, but load_source (data_staging.py ~536:
   `np.loadtxt(...)[:, param_cols]`), load_scalar_source (~779), and
   pool_size index loadtxt output as 2-D; NumPy returns (n_columns,)
   for one row -> IndexError (generator_core.py ~610 already uses
   np.atleast_2d — the idiom exists in-repo). Contract: normalize
   text tables to exact 2-D + validate column counts; a one-row val
   file stages normally; one-row training reaches the INTENTIONAL
   geometry/standardization verdict, never an incidental parse crash.
3. **Data controls whitelisted, not validated.** split_seed consumed
   through bare int() (fractional truncates, bool becomes int), never
   required/type-checked; ram_frac unchecked (`float(get(...))` —
   NaN/negative silently force streaming, > 1 or inf can force unsafe
   full materialization); param_cuts validates keys only (NaN bounds
   reject every row silently, inf erases the cut, bools act numeric,
   quoted values fail late).
4. **Contract:** one pure data-control validator in from_config
   before any staging — split_seed required, exact non-bool int in a
   stated range; ram_frac (when present) finite non-bool real in
   [0, 1]; every active cut bound finite non-bool real with lo < hi
   on paired bounds; NO coercion; shipped valid configs unchanged.

## Generator ingress identity (red-team 2026-07-12 fourth wave, Architect-VERIFIED, open)

train_args.ord is validated by SET equality only
(generator_core.py ~337), so duplicate names pass whenever the set
matches — and the two reorder helpers (~458-466) collapse the
duplicate DIFFERENTLY: yaml_to_ord maps the duplicated name twice
([0, 1, 2, 1] for ord [As, H0, omch2, H0]), ord_to_yaml's dict
comprehension keeps the LAST occurrence ([0, 3, 2]); the sampler runs
at dimension 4 against a model of dimension 3, and the chain/header/
sidecars can carry duplicate columns. The covariance-header pidx
(~326) has the same last-duplicate-wins defect.

Contract (Implementer unit; the red-team block of record adopted
whole): validate ord structurally (the one-list form, nonempty unique
string names); compare cardinality AND membership against cobaya's
sampled set, reporting missing/extra/duplicate separately; the covmat
header nonempty + unique before pidx; the loaded covariance finite,
2-D, square, aligned with the header length before subsetting (SPD
handling unchanged); fiducials finite numeric non-bool before
sampling; an MCMC thin/unique shortfall FAILS instead of warning and
publishing a smaller dataset under the requested identity;
run_generator rejects unparsed CLI arguments. Pure gates (no
CAMB/MPI): valid unique reorders preserve today's index maps;
duplicate/missing/extra/wrong-nesting/non-string ord raise with
separate diagnostics; duplicate header + header/matrix mismatch
raise; NaN/Inf covariance or fiducial raise before sampling;
insufficient unique rows cannot publish; an unknown optional flag is
rejected. The verbatim lensing physics loop is untouched.

## Generator scalar/grid configuration finiteness (red-team continuation — ADJUDICATED: folded into unit 17, generator ingress identity)

**Architect adjudication (Fable, at the merge):** VERIFIED and folded
into unit 17 as its finiteness extension — same ingress boundary, one
validator surface. Spot-confirmed anchors: dataset_generator_mps
_read_train_args coerces with int()/bool() (~110-139: int(seg[2]),
bool(seg[3]), bool(train_args["write_syren_base"]) — a quoted "false"
is True, 8.9 truncates to 8); generator_core ~659 uses
`math.isinf(logprior)` and isinf(NaN) is False, so a NaN cobaya prior
is accepted and written with the uniform lnp = 1. The contract below
is binding on unit 17's implementation.

The family generators validate ranges only after lossy coercion.
Executed probes against their real `_read_train_args` methods found:
MPS `n=8.9` / `nk=8.9` truncate to 8, quoted endpoint and
`write_syren_base: "false"` become true, and NaN `extrap_kmax` is
stored; CMB `lrange: [2.9, 500.9]` becomes `[2, 500]`; background
`nz=8.9` becomes 8. Validate the parsed values before conversion:
counts/multipoles exact non-bool ints, switches exact bool, every grid
edge/extrapolation limit finite, unknown family keys loud, and
`extrap_kmax >= max(k)` after finiteness. Valid shipped axes and model
requirements must remain byte-identical.

The shared post-sampling prior check uses only `math.isinf(logprior)`;
`math.isinf(NaN)` is false, so a uniform sample whose Cobaya prior is
NaN is accepted and written with the hard-coded uniform `lnp = 1`.
Before sampling require finite ordered bounds (and finite fiducial /
covariance / Cholesky / inverse on Gaussian runs); every prior result
uses `isfinite` in the log-posterior, post-sampling, append, and reload
paths. Nonfinite modeled columns or metadata fail before chain/dv
publication and name their parameter/row.

## The physical cuts (data.param_cuts)

- Schema: nested `data.param_cuts:` block, whitelist of 8 keys;
  `omegabh2_hi` REQUIRED; omegabh2_lo, omegam2h2_lo/hi, omegamh2_lo/
  hi, omegamh2ns_lo/hi optional (absent = that side open). Legacy flat
  keys / `omegabh2_cut` raise the paste-ready migration ValueError.
- The formulas (verbatim, strict inequalities, intersection-composed):
  obh2 = omegab*(H0/100)^2; g2 = (omegam*H0/100)^2 (= Gamma^2, the
  transfer shape parameter; Planck ~0.045); omh2 = omegam*(H0/100)^2
  (Planck ~0.143); omh2ns = omh2*ns (Planck ~0.138). Implementation is
  a QUANTITY TABLE (name, tag, needed cols, formula, lo, hi) — adding
  a window = one helper + one row, never an if-ladder. One-character
  traps (omegamh2 vs omegam2h2, ~3x scale apart) are why the whitelist
  is loud.
- THE FRAMING LESSON (the user's coverage argument beat the
  failure-targeted fit): n_train draws from the CUT pool, so cutting
  rarefied volume DENSIFIES the kept region instead of costing data —
  a coverage cut, not a failure cut. Evidence: sparsity and failure
  rate are U-shaped along omegam^2h^2 (the sampling cloud's long axis
  in (lnOm, lnh)); the (0.015, 0.08) window removed 96% of dchi2>100
  and took frac>0.2 from 0.59 to 0.156 same-day.
- Post-cut signature: residual failures HUG the cut boundaries
  (one-sided training support at the window edge) — the motivation for
  the generator's --boundary < 1.
- Live production values (2026-07-03, user-confirmed): omegabh2
  (0.005, 0.035), omegam2h2 (0.015, 0.08); omegamh2/omegamh2ns windows
  exist but ship commented out in the example YAML.
- The banner prints each active window's formula tag + per-window kept
  count (the only starvation diagnostic under stacked windows); the
  pool guard raises "physical pool too small after param_cuts" naming
  kept/requested and the remedy.

## Cut shading on the diagnostics triangle

Every active window shades sharp panels in ONE shared semi-transparent
grey `_CUT_GREY = (0.55, 0.55, 0.55, 0.30)` at zorder 0 (superposition
composes the union); sharp-only principle — no fuzzy projections onto
panels that do not determine a window; each plotting-side formula
carries a provenance comment naming phys_cut_idx's quantity-table
helper (ONE source of truth). D-GTB-1 lesson: the gate classifies the
shading layer by its design contract (zorder 0), never by rendering
heuristics.

## Generator physics execution: no silent zip truncation, no order-picked truth source (red-team 45M-06, 2026-07-12, Architect-VERIFIED; queue 33 — joins the file-set/ingress campaign, now 8+17+25+26+28+33)

Verified anchors: dataset_generator_lensing.py:103,
dataset_generator_cmb.py:307 AND :327 (two loops — the execute loop
and the read-results loop), dataset_generator_mps.py:364 all run
Cobaya components via
`for (x, _), z in zip(self.model._component_order.items(),
self.model._params_of_dependencies)` with z never used and
cached=True. zip stops at the shorter input, and both structures are
PRIVATE Cobaya internals (leading underscore, no stability promise) —
a length mismatch silently skips the remaining physics components,
with no count assertion anywhere. This exact hand-built lifecycle has
already failed once in this repo: dataset_generator_background.py
:321-335 records the bitwise-constant H(z) dump it produced and the
switch to the public model.logposterior(sample, cached=False)
lifecycle. The other three generators still run the failed pattern.
Separately, dataset_generator_lensing.py:99 picks its truth object by
YAML insertion order
(`self.model.likelihood[list(self.model.likelihood.keys())[0]]`) — a
dummy or auxiliary likelihood listed first silently changes the
data-vector producer without changing the requested probe.

Contract (Implementer):

1. Component execution is never zipped against an unused private
   list.
2. PREFERRED: the public logposterior(cached=False) lifecycle in all
   four generators — background is the worked reference. If a private
   component loop must remain for a demonstrated reason, validate the
   two structures' lengths and identities before iteration and fail
   loudly on any mismatch; silent truncation is forbidden.
3. Lensing selects the data-vector likelihood by an explicit unique
   capability/identity rule, not mapping order; zero or multiple
   candidates raise and list their names; the selected producer is
   validated against the requested probe.
4. The sampled-row reordering, name/value mapping, Cobaya input
   transformation, provider update, and physics evaluation split into
   named steps (the relevant alien-Python repair; ordinary tensor
   method chains stay allowed per the user ruling).
5. The producer that served each sample is recorded in generator
   provenance or the manifest, so a dump's truth source is auditable.
6. One physics evaluation per accepted sample; current output shapes
   and units preserved.

Gates:

- Fake-Cobaya pure legs under gates/checks/, registered on the board:
  a component/dependency length mismatch cannot skip a component;
  reversing likelihood insertion order does not change the selected
  producer; zero/multiple data-vector producers raise diagnostically;
  every intended component executes exactly once; a mutation arm
  reproduces the old truncated/private-loop form and proves the gate
  catches it.
- Workstation (Cobaya/CAMB): the lensing, CMB, and MPS generator
  smoke legs gain two distinct cosmologies; the generator payload is
  compared against the corresponding public-provider result for each,
  and the two payloads are asserted non-stale / non-identical.

## The old unstable variance survived beside the Chan accumulator (red-team 45M-23, 2026-07-12, Architect-REPRODUCED; queue 44 — amends the stable-moments standard, one numerical-statistics design repo-wide)

data_staging.py::stream_stats(method=1) (:113-129) still ships the
exact s1/s2 algorithm the grid2d revision prohibited: sum(x) and
sum(x^2) accumulators (:117-123) subtracted as
(s2 - s1*s1/n)/(n-1) (:128), with comments claiming float64 "avoid[s]
overflow" — the real hazard is catastrophic cancellation, not
overflow. Architect reproduction on the exec-extracted shipped body:
10,000 rows at offset 1e8 give std 1.8103 vs true 1.0017 (silently
wrong); offset 1e10 gives NaN (negative variance under the sqrt).
Scope, per the red team's own honest note: load_source keeps only
dv_mean and discards this std, so today's ordinary target centering
is NOT corrupted — the defect is a documented public normalization
function returning false or NaN scales, and two contradictory
numerical standards now living side by side.

Contract: method 1 reimplemented with the SAME float64 per-chunk
mean/M2 + Chan merge accepted for bounded grid2d staging (ddof=1
preserved); method validated exactly in {1, 2}; CHUNK a positive
non-bool integer; 2-D input; nonempty unique in-range indices;
method 1 needs >= 2 rows; nonfinite selected payloads and nonfinite
outputs rejected naming the column; the zero-scale policy explicit
(no division poison); docstring/comments corrected to name
Chan/Welford and the sample-variance convention. Red legs (torch
return -> an existing identity gate or a small board-listed staging
gate on the workstation): high-offset/small-spread truth vs
np.std(ddof=1); multiple chunk sizes and index orders; a fixture that
FAILS the s1/s2 formula (catch-power); empty/one-row/duplicate/
out-of-range indices; invalid method/chunk; nonfinite input; constant
column.

## A CMB dump has no multipole identity (red-team 45M-30, 2026-07-12, Architect-VERIFIED; queue 47 — CRITICAL, joins the file-set-authenticity cluster, now 8+17+25+26+28+33+47; sharpens the 45M-27 amendment to unit 26)

The CMB generator slices all four spectra by train_args.lrange
(dataset_generator_cmb.py:322-327, lsel) but writes only four
ANONYMOUS 2-D stores (_tt/_te/_ee/_pp.npy) — unlike the background
and MPS generators, no axis sidecar exists. Training checks ONLY
width: experiment.py:3549 `dv.shape[1] != ell.size` against the
covariance, then labels dump column 0 with the COVARIANCE's first
multipole. A dump generated for lrange [10, 1008] has the same 999
columns as a covariance for 2..1000: training accepts it and every
row, center, sigma, roughness stencil, convolution coordinate,
prediction, and saved artifact is consistently wrong with no shape
error. The checkpoint path is weaker still: the cmb _dv_load_chk
(:143-155) plus the core loader check 2-D shape/rows/RAM policy only
— no axis fact exists to compare, so a resumed run reuses a stale
same-width dump generated under a different multipole interval.

Contract (Implementer):
1. The generator publishes a CMB ell sidecar in the same file set —
   the exact integer column axis used to slice all four spectra.
2. Fresh allocation, checkpoint resume, append, and training all
   require exact equality among generator lrange, sidecar ell, every
   spectrum width, and covariance ell.
3. The axis must be the exact consecutive observable grid
   np.arange(lmin, lmax + 1), with the ruled production requirement
   lmin == 2 where applicable.
4. The sidecar is a REQUIRED member of the transactional checkpoint
   manifest (the unit-25 contract): missing, stale, noninteger,
   duplicated, gapped, shifted, or wrong-length axes fail before any
   row is trained.
5. The artifact persists the verified axis as today, but records
   that it was verified against the DUMP-side axis, not inferred
   from the covariance.
6. Science coordinates are never inferred from a filename or a
   width.

Red legs: valid generator -> checkpoint reload -> training chain
with one exact axis passes unchanged; same-width shifted dump vs
covariance (10..1008 vs 2..1000) raises; gapped/permuted/duplicated/
noninteger sidecar raises; missing sidecar on a checkpoint raises
before trusting any spectrum store; one of TT/TE/EE/PP with a width
inconsistent with the shared sidecar raises and leaves no accepted
checkpoint; the YAML lrange changed over old same-width files —
resume raises rather than relabeling columns; an anonymous legacy
CMB .npy with no axis fact raises with a migration instruction,
never guessing from covariance width. Relation to 45M-27 (unit 26
amendment) adopted as argued: validating the covariance's ell
sequence is necessary but insufficient — the dump must declare and
match its OWN sequence.

## UNIT 56 (45M-48): generators mark non-finite science payloads successful

Sixth 45M batch (2026-07-12), Architect-verified on HEAD. The shared
generator validates SAMPLING inputs, but no common boundary
validates the computed science payload before writing it and setting
failed[i] = False. Joins the file-set/ingress campaign — cluster now
8+17+25+26+28+33+56. Distinct from unit 17 (parameter/covariance/
config ingress) and unit 33 (component-execution truth): this unit
owns the payload boundary between the producer and the store. It is
NOT a separate publication system.

Verified chain: serial success path generator_core.py:908-921 sets
failed[i] = False inside the try and calls _dv_write OUTSIDE it, so
a broadcast-compatible wrong shape writes silently after the flag is
already cleared; MPI main receive :990-999 and drain :1048-1057 both
run kind != "err" -> blind _dv_write + failed = False. The default
store write :588-590 assigns blindly, as do the family overrides
(cmb :276-279, background :295-298, mps :333-336 — all per-quantity
`store[i] = dvs[q]`, numpy-broadcastable). Only the allocator's
FIRST payload receives family shape checks (cmb :247,
background :266). Producers close nothing: lensing casts to float32
with no finite check (dataset_generator_lensing.py:118-120); CMB
fills four spectra unchecked; background casts h/dm unchecked; MPS
checks only PRE-CAST pk_lin (dataset_generator_mps.py:390) — pk_nl,
boost, and both syren bases are unchecked, and a finite float64 can
overflow to Inf in the float32 cast AFTER the existing check.
Untruncated grep: the only payload-side isfinite in
compute_data_vectors/ is that mps :390 line.

Reproduction (numpy, Mac): float64 source [1, NaN, Inf, 1e100]
stores as float32 [1, nan, inf, inf] under the current success path
with failed = False. The 1e100 element is finite before conversion
and non-finite in the actual dump — validation must describe the
STORED payload, not the producer's pre-cast result. A scalar payload
broadcasts silently into a full row.

Contract:

1. One shared payload-validation boundary runs on the first payload
   and every subsequent serial/MPI result BEFORE _dv_write and
   BEFORE clearing the failure flag.
2. Validation happens after conversion to the exact storage dtype.
3. Exact family structure and shape: lensing one exact-width vector;
   CMB exact (4, nell); background exact keys h and dm, each
   matching its grid; MPS exact configured quantity keys, each
   matching nz*nk.
4. Every stored value finite. Legal signed quantities (CMB TE) are
   preserved; positivity rules remain family-specific.
5. MPS additionally requires finite, positive pk_lin, pk_nl, and
   boost, plus finite configured syren bases.
6. A bad payload follows the existing failed-row mechanism and can
   never be marked successful or published as usable training data.
   Final dataset closure (failed-row exclusion at staging + a
   nonzero exit) remains the wave-1 dataset-readiness unit's
   responsibility.

Gate legs (CPU-only; no torch or GPU): finite control; second-row
NaN and +-Inf; a finite float64 value overflowing to float32 Inf;
a wrong but broadcast-compatible later-row shape; a missing and an
extra mapping key; MPS finite pk_lin with NaN pk_nl or boost; a
legal negative-TE control; and the serial and MPI-result handlers
proven to invoke the IDENTICAL validator.

## UNIT 56 AMENDED (45M-48 addendum): broadcast relabeling + write-once/read-back

Seventh 45M batch (2026-07-12). The addendum's mechanism was already
reproduced during the unit's adjudication (a scalar payload
broadcasts silently into a full stored row); numpy row assignment is
not an exact-shape check, and the allocator's first-payload contract
protects nothing after row 0. Amendment to the contract:

1. The shared payload boundary requires, for every sample and every
   named family component, an EXACT key set and an EXACT predeclared
   shape before storage; scalar and length-one broadcasting are
   forbidden.
2. After the shape gate: cast to the real storage dtype, require
   every stored value finite, write ONCE, then read back (or
   validate the exact cast payload) before clearing the failed bit.
3. The family _dv_write methods are writers only — never independent
   validators.

Added red legs: first payload right-width, second payload length-one
-> the second sample is failed, never broadcast-successful; a
background/MPS dict omitting or adding one quantity key -> exact-key
refusal; a CMB later payload with a broadcast-compatible non-(4,
n_ell) shape -> refusal; serial and MPI result paths produce the
same failed/status verdict with no science row marked successful.

Rider REJECTED as factually absent (recorded so it is not
re-proposed): the claimed duplicate `self.datavectors[i] = dvs` in
the generic _dv_write does not exist on main or on the branch HEAD —
generator_core.py:588-590 contains exactly one assignment (verified
on both checkouts, 2026-07-12). No code change owed.

## UNIT 57 (45M-52): the generator reads its error capture before buffered writers publish

Seventh 45M batch (2026-07-12), Architect-verified and REPRODUCED
with the real repository function body. Joins the file-set/ingress
campaign — cluster now 8+17+25+26+28+33+56+57. Complementary to
unit 56, not subsumed by it: post-cast finiteness/shape validation
cannot catch a finite payload the solver itself declared invalid or
unconverged in text. Interlocks unit 33: if 33's preferred-path
harmonization replaces terminal parsing with a solver
status/exception API, that route satisfies this contract too.

Verified chain: capture_native_output (generator_core.py:163-185)
dup2-redirects fds 1/2 to a TemporaryFile with NO flush of Python,
C, or Fortran user-space buffers on entry; every family reads
tmp.read() INSIDE the with block (lensing :101-116, cmb :305-320,
background :333-341, mps :362-377) with no flush before the read;
the error-keyword scan then decides success and failed[i] is
cleared. MPI does not repair it — the worker makes the same
premature decision and sends "ok".

Reproduction (Mac, the exec'd real body, stdout block-buffered as in
a batch job): os.write(1, b"OS ERROR") IS captured; print("PYTHON
ERROR") captures EMPTY; libc.printf(b"C ERROR") captures EMPTY and
the text leaks to the RESTORED stdout after a later fflush; and the
un-flushed PYTHON ERROR text from one capture block appeared INSIDE
THE NEXT capture block together with pre-entry text — cross-sample
misattribution in BOTH directions (a clean sample can inherit its
predecessor's error text, and a failing sample's text can vanish
into a later row's verdict). A native component can therefore print
a declared-fatal string, return a finite-looking payload, and be
marked successful.

Contract (adopted):

1. Native-output synchronization is part of the correctness
   boundary, not a logging convenience. Flush Python streams before
   redirection (no earlier text misattributed); after the theory
   call, flush every supported writer before reading.
2. No generic Fortran/C capture claim unless the implementation
   proves the actual CAMB writer is synchronized. If in-process
   flushing cannot be proven reliable, isolate the solver behind a
   process boundary whose exit closes/flushes streams, or replace
   terminal parsing with a solver status/exception API carrying the
   same failure semantics.
3. Read the capture only after synchronization, scan once under a
   named documented case policy, and include the captured text in
   the raised error.
4. Restoration and temp-file cleanup stay exception-safe; a
   flush/read failure FAILS the sample, never silently downgrades
   the guard.
5. Serial and worker paths use the identical helper and verdict.

Red legs (pure CPU generator-core gate): immediate os.write caught;
buffered Python output caught without leaking to the restored
stream; buffered C printf caught; text emitted before entry not
attributed to the sample; an exception restores both descriptors; an
"ERROR"-emitting finite-payload fake is failed/zeroed in serial and
reported "err" by the worker decision helper. The C/Fortran leg must
exercise a genuinely buffered native writer — the current unflushed
body must FAIL it. If CAMB-specific flushing is the chosen design,
one small real-runtime leg, or an explicit refusal of unsupported
native runtimes rather than an advertised universal capture.

## UNIT 33 AMENDED (45M-64, fifteenth batch, 2026-07-12): the lifecycle verdict is the acceptance fact, not "the call returned"

Finding (red team, CONFIRMED live): the background generator's
repaired lifecycle is verdict-blind. _compute_dvs_from_sample
calls self.model.logposterior(sample[idx], cached=False) and
DISCARDS the returned object
(dataset_generator_background.py:335); success is then the
ABSENCE of {"ERROR", "error", "Did not converge"} in captured
terminal text, after which the provider getters run
unconditionally. Proven on REAL cobaya 3.6.2 (2026-07-12): a
rejecting component returns LogPosterior(logpost=-inf) as a
NORMAL return value — no raise, no keyword text — while the
explicit prior precheck stays finite (it covers only the prior,
not a theory/likelihood rejection). The reachable wrong result:
a rejected point leaves the provider holding the PREVIOUS
point's finite arrays; H and D_M are read and returned as a
successful payload; the shared writer records a success. This is
exactly the stale-physics class the cached=False repair was
built to close — disabling the cache forces recomputation when
computation HAPPENS, but does not validate that it succeeded.
Unit 56's payload boundary cannot distinguish those finite stale
values from new physics (its charter is the stored array, not
the execution). Entry 33's own text called background:335 "the
worked reference" for the three migrations — the worked
reference is the patient; it becomes the FIRST PATIENT of the
acceptance helper this amendment defines.

Contract amendment (the red team's seven clauses adopted, one
concretization):

1. Every generator captures the lifecycle's returned LogPosterior
   and requires a FINITE accepted logpost BEFORE reading any
   provider output. The verdict is the acceptance fact.
2. Acceptance uses cobaya's documented result API
   (LogPosterior.logpost; fields verified on cobaya 3.6.2:
   logpost/logpriors/loglikes/derived). Captured terminal text is
   DEMOTED to supplementary diagnostic evidence attached to the
   error report — never the verdict.
3. A rejected result invalidates the sample even when the
   provider contains finite values.
4. This-call provenance, PRECISION CONCRETIZATION (Architect):
   real cobaya exposes no provider generation token, so in the
   real path provenance is established by the CONJUNCTION
   (cached=False forced recomputation) AND (accepted finite
   verdict, checked BEFORE the first getter call) — a recorded
   derivation, not a new API; the fake-Cobaya gate proves the
   ORDER mechanically (legs below). Acceptance never rests on "a
   getter returned".
5. Unit 56's payload validator remains defense in depth AFTER
   lifecycle validation; a finite payload does not prove accepted
   execution.
6. ONE shared acceptance helper, executed identically by all four
   generator drivers — background immediately, lensing/CMB/MPS at
   their unit-33 migration — so the background omission is never
   copied three times.
7. Error reporting names the sample index, the parameter mapping,
   and the lifecycle verdict (logpost, and the rejecting
   component when cobaya reports one); no keyword scans as logic,
   no secret dumping.
8. Consequence, recorded: any background dump generated BEFORE
   this amendment lands is suspect for SPARSE stale rows (the
   bsn-smoke dump-variance tripwire catches only whole-dump
   constancy); science-grade dumps are regenerated after landing.

Red legs (in unit 33's fake-Cobaya, board-listed gate; no torch
needed):

- accepted finite lifecycle + fresh provider payload passes;
- rejected/non-finite lifecycle + stale finite provider payload
  fails BEFORE any getter or write — the instrumented fake
  provider PROVES zero getter calls were made;
- rejected lifecycle + fresh-looking finite payload still fails;
- accepted lifecycle + non-finite payload reaches and fails unit
  56's boundary (the defense-in-depth ordering leg);
- generation-token leg IN THE FAKE: the fake provider tags its
  arrays with the lifecycle call generation; the accepted leg
  asserts the arrays read belong to THIS call — the
  read-after-verdict discipline made mechanical;
- mutation arm: restore the discard-the-return / scan-text form —
  it must ACCEPT the stale payload, proving catch power;
- census leg: all four generator drivers execute the IDENTICAL
  acceptance helper once per sample.

Distinct from unit 56 (it validates the stored science array;
this amendment validates that the requested cosmology actually
produced it). Placement: rides unit 33 in the ingress cluster
(8+17+25+26+28+33), unchanged. USER-VISIBLE: rejected points now
fail their sample loudly with the verdict named (previously they
could write the previous cosmology's physics as a success).

## UNIT 8 EXTENDED (45M-68, seventeenth batch, 2026-07-12): the loader verifies parameter names, then ignores them and slices by position

Finding (red team, CONFIRMED live): load_source cross-checks the
GetDist .paramnames sidecar against the covariance header
(check_paramnames, data_staging.py:527-533), then DISCARDS the
resolved name list and selects columns with the positional
default param_cols = slice(2, -1) (:448, :536) — exactly two
leading bookkeeping columns, exactly one trailing derived column,
an assumption the check's own docstring bakes in ("e.g. chi2* —
which the staging slice already drops"). pool_size repeats the
literal slice (experiment.py:3303), so staging and pool
accounting share the defect instead of one checking the other.
Reproduced through the REAL load_source (2026-07-12): a valid
table [weight, lnp, a, b, d1*, d2*] with a fully-declaring
sidecar passes check_paramnames and returns C of width 3 (both
inputs plus the first derived value) against the two-name
covariance — the whitening dimension mismatch downstream; a
zero-derived table [weight, lnp, a, b] passes the same check and
returns C of width 1 — sampled parameter b silently DROPPED. The
sharpest fact is in-file: _scalar_columns (:632) ALREADY resolves
columns by name for load_scalar_source, and its docstring states
outright that "a fixed slice like load_source's slice(2, -1)
cannot locate them" — the scalar path does it right while the
main path guesses.

Contract (the red team's seven clauses adopted; the resolver
GENERALIZES _scalar_columns — no parallel mechanism):

1. ONE shared named-column resolver owns the parameter table for
   load_source, load_scalar_source, pool_size, checkpoint reload,
   and any generator readback.
2. It parses the COMPLETE .paramnames sequence including which
   entries are derived, and maps each covariance/header input
   name to its exact numeric column after the two GetDist
   bookkeeping columns.
3. Never infer "last column is chi2": derived-column count and
   placement come from the sidecar/manifest.
4. Exact uniqueness, presence, numeric table width, and order
   agreement are required before any value is selected.
5. The repository generator's current [weight, lnp, sampled...,
   chi2] form selects byte-for-byte identical values.
6. A missing sidecar follows ONE explicit documented legacy
   format contract or is refused with migration instructions —
   never an undocumented positional guess.
7. pool_size calls the same resolver and returns the same
   legal-row ceiling as stage_train.

Red legs (CPU):

- a current generator-shaped file selects the same sampled matrix
  exactly;
- zero, one, and multiple derived columns select the named
  sampled inputs correctly;
- a derived column interleaved among sampled names is selected
  correctly by the declared map, or refused if the documented
  format forbids the layout;
- duplicate/missing sidecar names, table-width mismatch, and
  covariance-name mismatch fail before staging;
- mutation arm: restore [:, 2:-1] — the zero-derived and
  two-derived legs must fail;
- pool_size and stage_train use the identical selected parameter
  matrix under every accepted layout;
- one-row versions compose with unit 11's exact-2D normalization
  contract (the 45M-65 amendment).

Distinct from the queued one-row normalization and no-cut
pool_size findings (row rank and optional cuts); this defect
chooses the wrong scientific COLUMNS on a multi-row finite table
whose sidecar declares everything correctly. Placement: unit 8
(checkpoint-set/file-set integrity), ingress cluster
8+17+25+26+28+33 — no new number. USER-VISIBLE: non-generator
GetDist tables now load their declared columns or refuse
(today: wrong columns, silently or via a downstream shape error).

## UNIT 33 AMENDED AGAIN (45M-70, seventeenth batch, 2026-07-12): the gate-side lifecycle calls have the same verdict blindness

Finding (red team, CONFIRMED by read at all six sites; the
rejection mechanism proven on real cobaya 3.6.2 in the 45M-64
adjudication): the three cobaya smoke gates and the CMB
covariance producer call model.logposterior(...), discard the
returned LogPosterior, and immediately read provider/theory
getters — cmb_smoke.py:441-442, bsn_smoke.py:248-252 and
:291-295, mps_smoke.py:331-335 and :376-380,
compute_cmb_covariance.py:420-422. None passes cached=False.
Since rejection is a NORMAL -inf return, a theory can populate
provider state, a downstream component can reject the point, and
the gate then compares the populated state against the reference
and greens: the gates prove "a value was readable", not "the
full lifecycle accepted the cosmology whose value they claim to
validate". The two-call forms can additionally bless a prior
accepted point's state after a rejected second evaluation.

Contract (the red team's eight clauses adopted, one ordering
ruling):

1. Every gate-side logposterior call captures the returned
   object.
2. Before the FIRST provider or theory getter, the point verdict
   is asserted accepted via the documented numeric fields — never
   exception absence or terminal-text matching.
3. cached=False whenever a gate intentionally evaluates multiple
   points.
4. A rejected point executes ZERO getters and reds the gate.
5. The rule applies to BOTH the emulator-provider and the
   CAMB-reference calls — an invalid reference lifecycle is never
   accepted as truth.
6. ONE Cobaya acceptance definition shared with the 45M-64
   generator helper. ORDERING RULING (Architect): whichever side
   lands first ESTABLISHES the shared definition — the gate-side
   fixes ride the wave-4 family gate visits, which come before
   the ingress cluster, so the gates will likely establish it and
   the generator migration consumes it; the Implementer proposes
   a home importable from both gates/ and compute_data_vectors/;
   two definitions never exist.
7. The existing numerical comparisons are preserved after the
   verdict guard.
8. The CMB covariance producer is amended through the same
   acceptance helper: its fiducial result is captured and
   validated before get_Cl or get_CAMBdata.

Red legs (in the existing board-listed cmb/bsn/mps smoke gates;
workstation torch/cobaya):

- real-cobaya accepted control: finite verdict, then getter, the
  existing numerical comparison unchanged;
- a rejecting likelihood after a theory populates state: current
  code reaches the getter and can green; repaired code reds
  BEFORE it;
- valid first point + rejected second point: no stale/current
  provider output is read for the rejected point;
- instrumented provider counts getter calls; rejection requires
  exactly zero;
- CAMB-reference rejection is red, never reference truth;
- mutation arm: discard the returned LogPosterior and proceed —
  the rejection leg must fail.

Placement: rides unit 33 (verdict truth), with the gate legs
executing at the wave-4 family gate visits and the covariance
producer at the 33 helper landing. USER-VISIBLE: smoke gates can
newly red on rejected points (previously green on readable stale
state).
## Continued red-team findings: staging must teach its row coordinates and count every host copy (2026-07-12)

### 45M-83: the row-coordinate comments are correct only after a reader already understands the implementation

`load_source` builds `src["dump_rows"]` beside `C`, `dv`, and `idx`, but its
comment begins with internal nouns (on-disk indices, staged rows, sibling
dump) and an expression split across comment lines.  The nearby comments for
`C_mean` and the grid2d exception assume the reader already knows the two
staging regimes, why resident staging changes the row-coordinate system, why
the base dump does not change with it, and why only training owns means.

Required documentation contract:

- Define the two coordinate systems before the dictionary: a global row is a
  row number in the original files; a local row is a row number in the compact
  resident copy.
- Walk one concrete example through both regimes.  If original rows
  `[9, 2, 9, 5]` are selected, show the sorted unique disk rows `[2, 5, 9]`,
  the resident arrays in that order, and the local coordinates that address
  them.  State which arrays are copies and which object remains a memmap.
- Define `dump_rows` as the coordinate list for a second file written in the
  same row order, then name grid2d's base dump as the consumer.  Do not use
  “sibling” before defining it.
- Explain `param_stats` and `stream_stats` separately.  The first statistic
  returned is the training-column center.  The unused second return is the
  scale, because the parameter covariance supplies the input scale on this
  path.  Validation never estimates its own center.
- Explain why grid2d defers the output moments: it first thins the k axis and
  transforms the raw surface into the target-law quantity, so a mean of the
  raw, unthinned file is a different statistic and is deliberately not
  computed.
- Correct the overgeneralization in the current comment: NumPy advanced
  indexing of a memmap materializes the requested result.  The source remains
  disk-backed because each requested block is materialized on demand, not
  because the indexed result is itself a memmap view.

Acceptance is documentation-only: no AST-with-docstrings-stripped code hash
changes.  A first-time reader must be able to label every index as global or
local and every array access as eager, lazy, view, or copy without consulting
the gate history.

### 45M-84: the host-RAM decision omits the parameter copy it creates

`stage_source` computes
`rows.size * dv.shape[1] * dv.dtype.itemsize`, compares that number with the
RAM allowance, and, on the resident branch, materializes both `dv[rows]` and
`C[rows]`.  The parameter copy is absent from the decision.  This is the same
shared-resource-accounting class as the GPU loader defect: a storage decision
must count every object whose lifetime it creates, even when one is usually
smaller than the other.

Required implementation contract:

- The predicted resident bytes equal the bytes of the stored representation
  of `dv[rows]` plus the bytes of the stored representation of `C[rows]`, with
  no count based on the source dtype when the stored copy changes dtype.
- The comparison and the verbose line report the same named components and
  the same total.
- The disk-backed branch remains lazy for the data-vector dump and does not
  claim that the already-eager parameter table is disk-backed.
- A CPU-only gate chooses a memory allowance between “dv alone fits” and
  “dv plus C fits.”  Current code must take the resident branch and the fixed
  code must take the disk-backed branch.
- Companion legs cover the exact-fit boundary, a resident control, a
  disk-backed control, and numerical/row-order identity across the two
  regimes.

The comment rewrite in 45M-83 and the accounting correction land together so
the code does not teach a byte formula that the implementation no longer
uses.

## Structured evidence map — gate contract anchors (45M-72 foundation)

The board's structured evidence map (`Gate.evidence`) pins each migrated
gate to a stable, runner-validated anchor in its home note; the mechanism
and the audited rollout are documented in `gates-and-board.md`. The two
generation-side gates anchor here:

<a id="gen-a-generator-seed"></a>
**generator-seed (GEN-A) — the dataset generator samples from an owned,
recorded RNG.** A required integer `--seed` owns a numpy Generator threaded
through the uniform parameter sampling, the emcee walker init and the
sampler's own moves, and the thinning subselection (no process-global
`np.random` draw remains); the seed is type-checked and written to the
chain header, and same-seed draws reproduce. Append-replay and
worker-invariance ride the workstation smoke gates.

<a id="srm-a-stage-ram"></a>
**stage-ram (SRM-A) — host-RAM staging counts every array and keeps the
seeded row order.** Two silent-divergence surfaces. (1) Accounting:
`stage_source` counts BOTH the parameter and target compact copies (each at
its own dtype and width) plus the reindex array, so a narrow-output dump
keeps the disk-backed branch when the two copies together exceed the budget.
(2) Seeded order: `idx` is the run's seeded selection order (a distinct,
generally unsorted permutation prefix); both branches must present those rows
in that one canonical order, because the training loop applies the same epoch
permutation to whichever index stage_source returns. The resident branch
returns `searchsorted(rows, idx)` — the local coordinates that walk the sorted
compact copy in selection order — not a plain `arange`; the disk branch
returns the global selection order unchanged. The gate drives the real
per-source loader (`_build_loaders_one`) in both storage regimes (resident
gather and disk stream), draws one shared epoch permutation, and requires the
executed parameters, targets, and minibatch membership and order to match
row-for-row against a selection-order anchor, with a mutation arm restoring
`arange` that must break the match. A duplicate row in the selection refuses
loudly (the selection is a unique permutation prefix by construction, so a
repeat is upstream corruption that would train one cosmology twice and skew
the normalization stats). Companion legs: resident / disk controls, an
unequal-dtype case, the duplicate refusal with a unique control, the exact-fit
boundary (need below / equal / above budget, strict less-than a deliberate
policy), the honest three-term banner
(`params + dv + idx = total`, the operator that held, the branch), the
dv-only-estimate mutation arm, and `dump_rows[idx_src]` recovering the global
selection order so a row-matched base dump lines up in either regime.

## UNIT 64 (BLOAT-02, RT-2026-07-13-01, 2026-07-13): one generator storage engine — the seven duplicated _dv_* operations move to generator_core

Finding (red team, CONFIRMED): dataset_generator_background.py (365
lines), dataset_generator_cmb.py (364), and dataset_generator_mps.py
(429) each carry the SAME seven storage operations by name and
structure — _dv_chk_files, _dv_load_chk, _dv_save, _dv_append,
_dv_alloc, _dv_write, _dv_zero — plus the _read_train_args frame;
roughly 449 shared lines maintained in triplicate, so a checkpoint or
append fix must be applied three times and can drift.
dataset_generator_lensing.py (136 lines) is thinner and must be
censused by the proposal (it may already delegate).

Contract:

1. PROPOSAL-FIRST (large-unit rule): the layout proposal is written
   in this note and audited by the Architect BEFORE implementation.
2. One multi-array store in compute_data_vectors/generator_core.py
   owning the seven operations, written as ordinary loops (C-readable;
   no comprehension cleverness) — the store handles N named arrays so
   the MPS multi-quantity case and the single-dv families use the
   same code.
3. Family physics (_compute_dvs_from_sample and friends) and sidecar
   generation STAY in the family drivers; the store owns bytes,
   checkpoints, and resume arithmetic only.
4. Acceptance is byte-identity: for all four generators, the dumps,
   checkpoints, and headers produced on the same seed + YAML are
   byte-identical before and after the refactor, including the
   append-and-restart path (a one-shot N+M equals fresh N then append
   M remains the separately-owed 45M-81 workstation leg, unchanged
   by this unit).
5. Sequencing: AFTER the generator-ingress cluster (units 8, 17, 25,
   26, 28, 33) — those units edit the same files, and consolidating
   under them would churn every open spec.

## UNIT 68 (RT-2026-07-13-06, 2026-07-13): optional latex metadata must not crash the run after sampling

Finding (red team, CONFIRMED by Architect probe on the real Cobaya):
a sampled parameter declared {prior: {min: 0, max: 1}} is valid and
yields model.info()['params'] WITHOUT a 'latex' key —
generator_core.py:808 evaluates param_info[x]['latex']
unconditionally, raising KeyError AFTER sampling completed and after
the chain .txt was written (:798-801). The run loses its
.paramnames / .covmat epilogue to a missing presentation string.

Contract (RULING): the parameter NAME becomes the GetDist display
label when 'latex' is absent — GetDist's own convention;
presentation metadata is NOT promoted to a required key and is NOT a
refusal surface. Legs: a real-Cobaya no-latex control runs the whole
epilogue and the emitted .paramnames is READ BACK by the getdist
reader (loadMCSamples), not just written; a with-latex control stays
byte-identical to today; the covmat epilogue executes in both.
Sequencing: rides the generator-ingress cluster (same file,
generator_core.py); no standalone landing.

## UNIT 56 — checkpoint-ingress amendment (20M-15, 2026-07-13, binding): resumed rows revalidate through the publication predicate

Finding (red team, CONFIRMED; shipped loader body executed with only
unavailable imports stubbed): a two-row checkpoint holding [NaN, 1] /
[Inf, 2] under a both-succeeded sidecar loads cleanly, prints
"Loaded models", schedules no recomputation, and republishes — the
loaders validate existence / row counts / rank / nbytes only
(generator_core.py:503-521, :540, :580-594, :619-640) and trust the
sidecar bits.

Amendment (binding, the unit's read-side half): (1) immediately
after every _dv_load_chk — before printing, before loadedfromchk,
before scheduling — every row whose failure bit is FALSE revalidates
through the EXACT family-specific stored-payload predicate used at
publication (unit 56's write-side), on the actual stored dtype,
exact expected shape, and family semantics (generic/lensing exact
vectors; all four CMB spectra with signed TE; the background-domain
rules; the MPS positivity/linear/nonlinear/boost/base rules); (2) a
failed row's documented zero placeholder stays legal; (3) an invalid
row under a false success bit is a checkpoint/sidecar inconsistency:
REFUSE with nonzero status, modify neither file, never silently flip
a bit; (4) manifests/digests remain provenance, never a substitute
for payload validity; (5) valid resume byte-identical.

Legs (ratified; CPU): valid rows load; NaN / +Inf / -Inf under a
success bit refuse; a dtype-overflow value refuses on readback;
every family's domain-invalid payload refuses (incl. nonpositive MPS
products where positivity is required); the zero placeholder stays
legal; refusal fires BEFORE "Loaded models" with both files
untouched; a mutation restoring the structural-only loader accepts
the corrupt row and schedules nothing (must red); a census leg
proves every loader calls the ONE shared predicate.

## UNIT 82 (20M-16, 2026-07-13): row authenticity — the published parameter table IS the computed one, bitwise, on every path

Finding (red team, CONFIRMED; mechanism verified in code): fresh
generation writes the float64 chain at nine decimal digits
(generator_core.py:798-801) and independently casts the computation
copy to float32 (:804), while append reloads the written chain
(:365-368) — decimal and binary rounding do not commute at
midpoint-adjacent values (witness: default_rng(0) uniform index
1582, one float32 ULP; ~700 per million draws), so a fresh run
computes science one representable cosmology away from the row it
publishes, finite and shape-correct throughout.

Contract (ratified): (1) ONE canonical published parameter table is
materialized exactly once, before any science call; (2) the rows
every _compute_dvs_from_sample receives are BITWISE identical to the
rows the real training loader recovers from the published chain;
(3) fresh, resume, append, serial, and MPI share that representation
and row order; (4) never write float64 xf and independently cast a
second copy — either write a representation that round-trips exactly
to the canonical computation dtype, or write first and reload once
through the shared reader before any computation; the owner and the
conversion are explicit; (5) lnp, chi2, bound checks, and other
row-labelled facts are recomputed at the canonical row OR explicitly
stored as pre-canonical sampling diagnostics — never facts about a
different row; (6) the checkpoint/artifact manifest records the
canonical parameter dtype and representation; (7) valid values and
row order otherwise unchanged.

Legs (ratified; CPU): the seed-0/index-1582 witness through fresh
generation with an identity producer; chain read back through the
real load_source with bitwise equality among producer row, staged
row, and identity payload; an away-from-boundary control; fresh ==
append == resume on the same canonical rows; serial == MPI;
multi-parameter rows with one boundary-crossing coordinate; a
mutation restoring write-then-independent-cast must fail on the
witness; a census proving no producer call receives any table but
the canonical one. Placement: the generator-ingress campaign
(rides with the ingress cluster; distinct from 20M-15's payload
validity — 15 asks whether the stored answer is legal, 82 whether
it belongs to the printed cosmology).

## 45M-81 AMENDED (2026-07-13, CONFIRMED CURRENT FAILURE): append restarts the RNG stream and duplicates the original dataset — RNG state is checkpoint state

Finding (red team; Architect re-executed the repro exactly): fresh
generation with --seed S followed by --loadchk 1 --append 1 --seed S
reconstructs default_rng(S) from the beginning
(generator_core.py:270), the checkpoint restores rows/payloads but
NO generator state or draw offset, and the append branch calls the
same rng.uniform (:748): at the public minimum N = M = 200 every
appended row duplicates an original, fresh+append != one-shot, and
the first appended row is draw 1, not draw N+1. The :792 chain
header ("seed=... rng=numpy.default_rng") implies seed-only
provenance — false for append.

Binding amendment (converts the owed acceptance proof into this
contract): (1) the COMPLETE owned RNG state is persisted
transactionally with checkpoint generation — never the seed alone;
(2) append restores that exact state before any draw, OR
replays/advances only under a manifest-proven algorithm + prior-row
count with the resulting state VERIFIED; (3) every sampling engine
that owns state is covered, including the emcee move/random state on
the Gaussian branch (the queue-5 sampler._random rider joins here:
the private-attr assignment gains its attribute-presence assert and
its state joins the persisted block); (4) one-shot N+M and
fresh-N-then-append-M produce the SAME canonical parameter rows and
row order; (5) missing, stale, or mismatched RNG state REFUSES
append before modifying the old dataset; (6) the chain header stops
implying seed-only replay provenance; (7) MPI worker count must not
affect the parameter table (science-row ordering remains unit 82's
coordinated half). CAUTION: no dataset may be produced via append
until this lands; unit 82 is orthogonal (it would faithfully
canonicalize the duplicates).

Legs (CPU now): real uniform fresh/append restart; exact equality to
one-shot; no cross-half duplicates; state readback; missing/corrupt
state refusal; a mutation reinitializing default_rng(seed) must
reproduce the 200/200 duplication and red. The Gaussian/emcee
continuation and worker-invariance legs may stay workstation-bound
where their dependencies require it. Placement: the
generator-ingress campaign beside unit 82; blocks any production
append.

## UNIT 87 (20M-21, 2026-07-13): the chain's reserved column carries minus_logpost — named honestly, exact against chi2*, uniform mode never fakes a posterior

Finding (red team, CONFIRMED; Architect live GetDist probe
reproduces the reversal): the generator writes raw lnp into chain
column 2 on fresh AND append (generator_core.py:798, :844), which
GetDist defines as -log(posterior) and minimizes for the best fit —
row ranking, shading, and cooling all reverse. The trailing
chi2* = -2*lnp has the right sign but does not repair the reserved
column. Uniform mode fabricates lnp = 1 / chi2* = -2 (:751, :760),
which must never be described as an evaluated posterior.

Contract (ratified): (1) the in-memory quantity is named logpost;
minus_logpost = -logpost is materialized at the PUBLICATION
boundary; (2) minus_logpost is written into reserved column 2 on
fresh and append; (3) the exact relation minus_logpost == chi2*/2
holds when chi2* means -2 log p; (4) headers/sidecars are honest —
"lnp" may not label a negative-log-posterior column, and "chi2" may
not silently mean posterior; (5) the uniform-sampling record is
explicit: no evaluated posterior means the GetDist-required neutral
value with a declared unavailable status, or the derived diagnostic
omitted under a documented format — never -2 published as measured
chi-squared; (6) old sign-reversed chains carry an explicit
migration marker/tool or refuse — the convention is never inferred
from numeric values; (7) parameter rows and science payloads
unchanged.

Legs (ratified; CPU, real GetDist): the [-1, -10] known-answer pair
selects the -1 row; the current-sign mutation selects the worse row
and must red; fresh and append read back one convention; column 2 ==
trailing chi2*/2 exactly under the declared rounding policy;
headers/sidecars distinguish posterior from likelihood; uniform mode
claims no measured posterior; legacy missing-convention and migrated
chains covered. Placement: the generator publication/provenance
campaign (with the 45M-81 RNG amendment and units 68 + 82);
production Gaussian/MCMC generation blocked on it alongside them.

## UNIT 8 AMENDED (20M-23, 2026-07-13): the run-control state machine validates before any path mutation — an illegal append pair cannot touch the prior dataset

Finding (red team, CONFIRMED live; the constraint is in the CLI help
text and enforced nowhere): --append 1 --loadchk 0 parses cleanly
(generator_core.py:139-154), the flags are copied independently
(:238, :255), __load_chk returns False whenever loadchk != 1, and
:699 routes to the FRESH branch — np.savetxt replaces the existing
chain and the family path allocates fresh storage: the sentinel
dataset was destroyed by an accepted command.

Amendment (binding; extends the unit-8 dataset-transaction/manifest
campaign, no new mechanism): (1) the run-control state machine
validates BEFORE any output directory or file is created:
append == 1 requires loadchk == 1; (2) a legal append additionally
requires the manifest-authenticated prior generation unit 8 already
specifies; (3) the illegal pair raises a teaching error naming both
values and explaining that append EXTENDS the validated prior
dataset — it is not fresh generation at the same path; (4) rejection
preserves every byte and timestamp of the existing chain, fail
flags, data-vector members, axes, parameter-name sidecar,
covariance, ranges, and manifest; (5) fresh 0/0, resume 1/0, and
authenticated append 1/1 are the exhaustive legal states; any other
programmatic non-bool/non-integer state refuses before path
mutation; (6) 45M-81's RNG-state continuation stays an INDEPENDENT
requirement of legal append (authenticate the old dataset AND
continue the stochastic stream).

Legs (ratified; CPU): seed a complete sentinel bundle, drive the
REAL public CLI, assert the illegal pair raises with ALL member
digests identical; controls for the three legal states; a mutation
restoring independent flag handling must visibly destroy the
sentinel and red. Placement: the generator publication/provenance
cluster (with the 45M-81 amendment and units 68/82/87); production
generation blocked on the cluster.
