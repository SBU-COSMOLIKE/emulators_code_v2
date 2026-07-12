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

### Family sidecar paths and validation axes are not bound (red-team continuation, evidenced 2026-07-12; awaiting Architect adjudication)

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

## Generator scalar/grid configuration finiteness (red-team continuation, evidenced 2026-07-12; awaiting Architect adjudication)

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
