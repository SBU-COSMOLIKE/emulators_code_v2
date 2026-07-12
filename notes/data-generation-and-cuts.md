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
