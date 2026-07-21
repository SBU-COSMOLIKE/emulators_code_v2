# Data generation, staging, and physical parameter cuts

This note is the durable engineering contract for dataset generation,
checkpointing, staging, and physical-window selection. It is not a development
diary. A future change should update the rule that changed, the reason for the
rule, the code that owns it, and the evidence that proves it.

The user-facing introduction belongs in the repository README. This note keeps
the deeper rules needed by maintainers and automated development tools.

## Terms used throughout this note

A **checkpoint** is saved progress from an incomplete generator run.
**Staging** selects, validates, and places saved rows into arrays used by
training. A **sidecar** is a small companion file that records names, order,
axes, or other facts about a larger data file. A **gate** is a named
validation job whose required result is written before it starts.

A **digest** is a fixed-size fingerprint calculated from exact bytes. A
**probe** names the requested physical output, such as background or CMB. A
family **variant** selects one documented representation, such as native or
Syren-base matter power.

Two identities carry a dataset into a saved emulator. **Staged-selection
identity** records which source rows a training run staged and in what order
(an order-sensitive digest saved with the run's configuration).
**Artifact identity** then binds that staged selection to the resolved model,
training recipe, composition, and saved weights. The artifact note owns the
second identity. A matching filename or array shape proves neither of them.

The **maximum-correlation policy** limits the largest absolute off-diagonal
correlation in the covariance used by Gaussian Markov-chain Monte Carlo
sampling. Gaussian mode records a finite limit greater than `0.01` and no
greater than `1.0`. Uniform mode does not use a covariance correlation limit
and must record `null`. This mode-dependent rule is the policy's
**applicability**.

## Ownership map

Message Passing Interface (MPI) lets separate generator processes coordinate
work. The cosmic microwave background (CMB) is the relic radiation from the
early universe measured by the CMB family.

| Subject | Primary owner |
|---|---|
| Shared command line, sampling, MPI, and checkpoints | `compute_data_vectors/generator_core.py` |
| Family-specific physics and family sidecars | `compute_data_vectors/dataset_generator_*.py` |
| CMB covariance generation | `compute_data_vectors/compute_cmb_covariance.py` |
| Project path resolution for training configs | `emulator/cocoa.py` |
| Parameter-table schema | `emulator/parameter_table.py` |
| Row selection and host-memory staging | `emulator/data_staging.py` |
| Family staging and pool-size decisions | `emulator/experiment.py` |
| Physical-window formulas | the shared parameter-cut quantity table |
| Gate claims | `ai/gates/checks/` and the stable anchors in this note |

## Generator families

### Rule

`generator_core.py` owns the common command line, the two sampling modes, MPI
coordination, checkpoint control, and native-output capture. Thin family
drivers own only family physics and family-specific files:

- lensing writes the CosmoLike data vector;
- CMB writes temperature-temperature (TT), temperature and E-mode
  polarization cross-spectrum (TE), E-mode polarization auto-spectrum (EE),
  and lensing-potential auto-spectrum (PP) arrays. It also writes the
  multipole identity required by the dataset contract;
- background writes Hubble and transverse-comoving-distance arrays with their
  redshift axes;
- matter power writes linear power, boost, redshift and wavenumber axes, and
  both base arrays when the Syren-base variant is requested. Syren is the
  analytic matter-power baseline that this project can combine with an
  emulator correction.

`compute_cmb_covariance.py` is a separate producer. Its `params` block uses
the script's own names, including `omegabh2` and `omegach2`, and contains plain
resolved numbers.

### CMB covariance output

#### Rule

The CMB covariance archive is a generated scientific dataset, not an emulator
artifact. `compute_cmb_covariance.py` refuses when the requested output file
already exists, because an emulator may already be trained against it and a
silent replacement would change what that emulator's saved whitening meant.
Every persisted array must be finite before anything is written; the write
goes to a temporary name in the same folder and is renamed into place, so a
crash mid-write cannot leave a truncated archive under the public name.

#### Why

A covariance calculation can take substantial time, and training trusts the
archive bytes. A partial or silently replaced archive corrupts every later
training run that reads it.

### Generator evaluation boundaries

The background generator evaluates each point with the standard Cobaya
`model.logposterior(point, cached=False)` lifecycle. Other generators may keep
a different lifecycle only when their gates prove that every intended
component executes once and that every dumped row belongs to the requested
cosmology. A private zipped component loop may never truncate silently.

The matter-power-spectrum (MPS) `Pk_grid` request uses
`k_max = max(2 * max(requested_k), 20)`. Every dumped wavenumber must be
computed rather than extrapolated. The served interpolator may have a
separate, validated extrapolation limit.

### Why

The shared core prevents checkpoint and output-file rules from drifting among
families. Family physics remains explicit so changes to one scientific product
do not hide inside a generic storage abstraction. Lifecycle rules exist because
a finite provider cache can contain values from an earlier cosmology after a
later cosmology is rejected.

### Acceptance evidence

- Every family emits its exact required member set and no undeclared member.
- Two distinct cosmologies produce distinct, row-matched payloads when the
  underlying physics differs.
- The lifecycle result is accepted before any provider getter is called.
- A rejected lifecycle causes zero getter calls and cannot publish a row.
- The public-provider result and the generator payload agree for each tested
  cosmology.
- A mutation that restores silent zip truncation or discards the lifecycle
  verdict must fail.

## Sampling

### Rule

Fresh tempered sampling uses the resolved parameter covariance and an owned
random number generator. Fresh uniform sampling moves each requested endpoint
one representable floating-point value toward the interval interior before
drawing, so a draw can never sit exactly on a hard prior edge. The movement is
defined in float representation steps, so it is correct for endpoints of any
sign or magnitude, including zero; no boundary policy may depend on distance
from zero. The resolved (moved) bounds are what the `.ranges` file and the
scientific record publish.

Append draws its new rows from a stream derived from the seed together with
the existing row count. A stream derived from the bare seed alone would repeat
the original run's draws exactly and duplicate every existing row.

### Why

Uniform support changes when infinite prior endpoints are resolved with a
temperature, and a relative-factor endpoint shrink is wrong at and near zero.
Seed-only append restarts the random stream and can duplicate the original
dataset.

### Acceptance evidence

- Same-width intervals translated along the number line retain the same
  fractional interior width.
- Narrow intervals either have a representable interior or refuse before
  output mutation.
- Append on a checkpoint of `N` rows produces rows distinct from the first
  `N`, and rerunning the same append reproduces the same appended rows.
- A mutation that restores endpoint multiplication or `default_rng(seed)` at
  append must fail.

<a id="generator-seed-owned-rng"></a>
### Owned random-number generation

#### Rule

A required native integer seed initializes an owned NumPy generator for
parameter draws, Gaussian walker initialization, and unique-row selection.
The emcee sampler receives a separate random state seeded from that owned
generator, so the walk itself is replayable. Process-global `np.random` draws
are not part of the generator contract. The seed and generator name are
recorded in the chain header. Append derives its stream from the seed plus
the existing row count, as the sampling section above states.

#### Acceptance evidence

- Equal seed and equal request produce equal rows.
- A different seed changes the selected rows.
- Worker count does not change the parameter table (sampling happens on rank
  zero only).
- A scan and mutation check prove that no process-global random draw enters the
  generation path (the `generator-seed` gate).

## Canonical parameter rows

### Rule

One canonical parameter representation is materialized before any science
evaluation. Every family producer receives rows bitwise equal to the rows the
training loader later recovers from the saved table (the `%.9e` chain format
round-trips the generator's float32 rows exactly). Fresh serial and MPI paths
share that representation and row order. Resume reads the saved rows back;
append extends them with rows from the derived stream.

In MPI mode, rank zero records which row it assigned to each worker. A result
may change a payload store only when it comes from a worker with a live
assignment and reports that exact row. An unknown worker, duplicate result,
stale result, malformed message, or different reported row refuses before the
assignment is removed or any row is written. A worker's stop acknowledgement
must likewise come from a worker awaiting shutdown and name that same rank.

The public table uses GetDist's reserved columns honestly. GetDist is the
chain-table format and reader used for saved sampling results. Column two
stores `minus_logpost`, not raw log posterior. When the trailing field represents
`chi2* = -2 log p`, `minus_logpost == chi2*/2` under the declared rounding
policy. Uniform generation must not claim to have measured a posterior that it
did not evaluate.

`.ranges`, the chain table, and the parameter reader use one decimal or exact
representation policy derived from the owned dtype. A valid interval must not
collapse during serialization.

### Why

Writing float64 decimal rows while computing from an independently cast
float32 copy can move a coordinate by one float32 unit. The resulting data
vector then belongs to a different cosmology than the printed row. Low-
precision range output can also collapse two distinct bounds.

### Acceptance evidence

- A midpoint-adjacent witness is bitwise equal at producer input, table
  readback, and staging.
- Fresh serial and MPI paths agree on canonical rows, and resume preserves
  the saved rows exactly.
- Scripted MPI replies prove that a valid out-of-order result reaches only its
  assigned row, while a wrong row, inactive worker, duplicate reply, malformed
  reply, and false stop acknowledgement change no stored row.
- GetDist selects the higher-posterior row in a two-row known answer.
- The reserved-column sign mutation selects the wrong row and must fail.
- Float32-distinct range-bound pairs serialize and read back as distinct,
  ordered bounds containing all canonical rows.

## Dataset readiness

### Rule

A row's failure flag is cleared only after the family computation returned a
complete payload and that payload was written to the stores. A failed or
timed-out row keeps its flag set and a zero-filled payload. The failure flags
are saved beside the dump as the failfile (one ASCII `0` or `1` per row), in
the same checkpoint save as the payload stores, so the flags and the rows
they describe cannot drift apart.

Training must never treat a failed row's zero-filled payload as scientific
data: the training YAML names the failfile explicitly
(`data.train_failure_mask` / `data.val_failure_mask`) and staging excludes
the flagged rows before cuts and selection.

`--boundary` is a finite real with `0 < boundary <= 1`; an invalid value
refuses before any output path is opened, never silently becomes `1`.

### Why

A provider failure can leave a zero-filled row while the program exits zero.
Zero rows poison the training whitening and the learned function. An
out-of-range boundary silently coerced to `1` would generate a test/val dump
on the full interval the flag was supposed to trim.

### Acceptance evidence

- A failed sample leaves a set flag and a zero row; a successful sample
  clears the flag and stores its payload.
- Staging refuses a missing, short, or malformed failure mask and cannot
  satisfy a requested training size with a failed row.
- Boundary controls cover zero, negative, above one, nonfinite, and a valid
  interior value.

## Native-output capture

### Rule

Fortran and C output from the solver is captured at the file-descriptor
level around each sample's evaluation, and known error keywords in that text
fail the sample. Capture cleanup and descriptor restoration are
exception-safe. The background generator additionally requires a finite
accepted Cobaya log-posterior before reading any provider result, because a
rejected point returns `-inf` without raising or printing anything.

### Why

CAMB reports some failures only as Fortran text on the raw descriptors, where
Python-level redirection cannot see them. A rejected Cobaya point need not
raise; the provider can still contain a finite array from an earlier accepted
point.

### Acceptance evidence

- Native writes during a sample are captured and scanned; exceptions restore
  both descriptors.
- A background point rejected by the prior or the theory cannot clear its
  failure flag.

## Run controls and checkpoints

### Rule

`--loadchk`, `--append`, and `--chain` are `0` or `1`. The legal
`(loadchk, append)` pairs are exactly `(0,0)` for fresh generation, `(1,0)`
for resume, and `(1,1)` for append; `append=1, loadchk=0` refuses with an
error naming both flags, because append extends a validated prior dataset and
never means fresh generation at an existing path. `--chain 1` independently
selects a chain-only run.

A requested load (`--loadchk 1`) whose checkpoint files are missing refuses,
naming every missing file. A failed load never silently becomes fresh
generation: a mistyped path would otherwise regenerate, and overwrite,
instead of resuming.

Resume and append require the run's complete file set before trusting a
checkpoint: the chain, `.covmat`, and `.ranges` for every run, plus the
failfile, every family data-vector store, and every axis sidecar for a full
run. Loaded stores must match the configuration: row counts against the
chain, column counts against the configured grid or multipole range, and
every axis sidecar exactly equal to the coordinates `train_args` resolves to.
A checkpoint written for one grid must never be continued on another, because
the row payloads would silently mean different coordinates.

### Chain-only isolation

A `--chain 1` run writes only the parameter-side files, and all three output
stems carry a `_chain_only` suffix. A chain-only run therefore can never
overwrite, or resume from, the failure and data-vector files of a full run
that used the same names; chain-only resume and append read and extend only
the parameter-side files.

### Why

A chain-only run at full-dataset stems could replace parameter rows while
leaving old payload rows and failure flags, creating a same-shaped but
scientifically false pairing. An axis mismatch on resume is the same failure
inside one dataset.

### Acceptance evidence

- Boolean-invalid and illegal flag pairs refuse before setup.
- A load request with missing files refuses naming them; the refusal changes
  no existing file.
- Changed background/MPS coordinates, a changed CMB multipole range, and
  short stores refuse on resume; matching controls pass.
- Full and chain-only stems are distinct; chain-only paths perform no
  failure, payload, or axis input/output.

## Plain-file dataset outputs

### Rule

A generator run writes ordinary files into the project `chains/` folder, and
those files are the complete dataset:

- the parameter side: `<paramfile stem>.1.txt` (the chain),
  `.paramnames`, `.ranges`, `.covmat`, and the `.facts.yaml` scientific
  record;
- the payload side: one or more 2-D `.npy` stores named
  `<datavsfile stem>_<member>.npy`, with axis sidecars
  (`_z.npy`, `_k.npy`, `_ell.npy`) written once beside them;
- the failfile `<failfile stem>.txt`, one `0`/`1` line per row.

Stems follow `<name>_<probe>_<temp|unifs>` with the optional `_chain_only`
suffix. There is no publication layer, no locator, and no hidden state: the
files a run names are the files it writes, checkpoint saves go through a
temporary file and an atomic rename, and a training YAML names these same
files directly (resolved against the project `chains/` folder by
`emulator/cocoa.py`).

The `.facts.yaml` sidecar (format owned by `emulator/fixed_facts.py`) records
the cosmology the run held fixed, the dark-energy law in canonical `(w, wa)`
form, and the requested and resolved sampling supports, keyed by the digest
of the chain file it sits beside. The generator writes it whenever it writes
the chain; training staging refuses a dump without it.

### Why

Plain files keep the dataset inspectable with ordinary tools and keep the
generator understandable line by line. The scientific record and the failure
mask carry the two facts a payload file cannot: which cosmology produced the
rows, and which rows must not train.

### Acceptance evidence

- Each family smoke gate runs its generator and finds every named output
  file, including the axis sidecars and the scientific record.
- The dump-variance tripwire in `bsn-smoke` fails if every row carries the
  same background (the stale-cache failure the background lifecycle rule
  prevents).
- Checkpoint interruption and resume produce the same files as one
  uninterrupted run.

## Parameter-table schema

<a id="parameter-table-schema-and-layout"></a>
### Schema and layout

#### Rule

`emulator/parameter_table.py` is the only authority for parameter-table
columns. It resolves the exact stem first and a purely numeric chain-root stem
second. Nonnumeric dotted stems are not stripped. Every nonblank `.paramnames`
declaration is retained with its derived marker and numeric column after the
two GetDist bookkeeping columns.

Normalized names and requested inputs and outputs are unique. Requested names
exist. The complete nonderived sequence equals the requested input sequence,
including order. Requested outputs are derived. Numeric width equals two plus
the declaration count. Returned arrays are float32 and exactly two-dimensional,
including one-row and zero-output cases. A UTF-8 byte-order mark is transport
syntax; repeated `*` and `?` refuse. Missing metadata refuses with candidate
paths and migration guidance. There is no positional fallback.

#### Acceptance evidence

Current generator layout, zero derived columns, multiple interleaved derived
columns, one row, and zero outputs pass. Duplicate, missing, extra, reordered,
wrong-marker, and short or long table cases refuse. Restoring `[:, 2:-1]` or a
missing-sidecar guess must fail.

<a id="parameter-table-pre-dv-refusal"></a>
### Parameter refusal precedes payload I/O

#### Rule

`load_source` directly resolves the parameter table before opening a data-
vector member or staging data. The returned named input array is the source of
`C`, the parameter-input matrix whose rows are cosmologies and whose columns
follow the declared parameter names.

#### Acceptance evidence

Missing and invalid declarations leave payload-open and staging sentinels
untouched. Hidden earlier evaluation, wrapped right-hand sides, ignored
resolver output, extra payload opens, and pre-resolution staging mutations
must fail.

<a id="parameter-table-stage-pool-parity"></a>
### Staging and pool parity

#### Rule

Ordinary staging, scalar staging, pool sizing, and generator checkpoint
readback call the same resolver. Scalar and pool paths include requested output
names so an invalid target declaration is not counted as stageable. Pool size
and `stage_train` apply the same named columns and physical cuts.

No-cut configs are valid for scalar, CMB, grid, and grid2d. Their pool is the
full named table. Active cuts produce the same independently known survivor
set in pool sizing and staging. `pool + 1` refuses before training or grid2d
transformation.

#### Acceptance evidence

Each consumer is mutated separately while siblings remain correct. Every
isolated positional or required-cut-key mutation fails. Grid2d keeps the real
named loader and disk-backed source in the witness.

<a id="failed-row-staging-and-selection-identity"></a>
### Failed rows and exact staged selection

#### Rule

A full data-vector source requires the generator's failure mask, named
explicitly in the training YAML (`data.train_failure_mask` /
`data.val_failure_mask`). The mask contains exactly one ASCII `0` or `1` line
for every parameter and payload row. A missing mask, another token, or a
different row count refuses. Chain-only scalar data has no data-vector
failure mask. Staging validates the tokens and row count; the user owns the
choice of which mask pairs with which dump.

Staging removes failed rows before physical cuts, seeded selection, and
pool-size reporting. It cannot satisfy a requested training size with a
failed row.

After staging, the experiment records the staged selection in the
configuration saved with the emulator (`data._staged_selection`, one record
per split): the original source row count, split seed, physical cuts,
selected count, and an order-sensitive SHA-256 of the exact disk rows
addressed by the loader. For a compact resident source, that order is
`dump_rows[idx]`. For a disk-backed source, `idx` already holds global disk
rows. The recorder derives the applicable order from the staged array sizes
and refuses unless it equals `selected_rows` exactly. The records are saved
with the run's configuration, so an artifact states which rows trained it.

#### Acceptance evidence

- A four-row source with mask `0, 1, 0, 1` can stage only rows zero and two.
- Missing, short, malformed, and failed-row masks refuse before training.
- Resident, disk-backed, scalar, and all-row selections save the disk order
  that the loader actually addresses.
- Equal counts with a different permutation refuse before the fingerprint is
  saved.
- Both saved configuration records retain the staged row-selection
  identity.

## Host-memory staging

### Rule

Parameter rows are eager in memory. Data-vector dumps remain memory mapped
unless a compact resident selection fits the allowed memory. Two coordinate
systems are explicit:

- a global row is a row in the original files;
- a local row is a row in a compact resident copy.

If selected global rows are `[9, 2, 9, 5]`, uniqueness validation rejects the
repeat. For the legal unique example `[9, 2, 5]`, sorted compact storage is
`[2, 5, 9]`, while local coordinates `[2, 0, 1]` preserve the seeded selection
order. `dump_rows` records the global rows for a second row-aligned file such
as the grid2d base dump.

The resident-byte prediction counts the parameter copy, target copy, and index
array using their stored dtypes and widths. The diagnostic line reports the
same components, total, comparison operator, and chosen branch.

<a id="stage-ram-both-copies"></a>
### Both copies and seeded order

#### Rule

Resident staging requires the combined parameter, target, and reindex bytes to
fit. The strict exact-fit policy is explicit. Resident and disk-backed branches
present the same seeded row order to the training loader. A duplicate selected
row is upstream corruption and refuses.

#### Acceptance evidence

- A budget between target-only bytes and complete bytes selects disk-backed
  staging.
- Below, equal, and above-budget controls pin the comparison policy.
- Resident and disk-backed loaders produce identical parameters, targets,
  minibatch membership, and order under one epoch permutation.
- `dump_rows[idx]` recovers the same global order in both regimes.
- A mutation returning plain `arange` or counting only target bytes fails.

## Grid2d bounded staging

### Rule

Grid2d staging chooses retained `(z,k)` columns before reading any payload.
It reads only those columns in bounded row chunks from raw and base memmaps,
checks positivity and shapes per chunk, performs the law transform, and writes
stored float32 rows to a resident array or temporary memmap according to the
same memory policy. It never materializes an unthinned selected matrix.

Means and population standard deviations are accumulated from the exact stored
float32 payload promoted to float64. The stable accumulator merges
`(count, mean, M2)` with the Chan/Welford formula. Here `M2` is the running sum
of squared deviations from the current mean; after all values are included,
`M2 = sum((x - mean)^2)`. The accumulator never subtracts `sum(x)^2` from
`sum(x^2)`. Raw and base row counts and widths match exactly. The original
seeded selection order survives the grid2d law transform.

Train and validation temporary files have independent owners. Restaging
releases the superseded file. Explicit release functions support sweep cleanup.
Any transform failure unlinks a partial file; process-exit cleanup is only a
fallback.

### Why

Production grid2d arrays are too large for whole-selection float64 copies.
Naive variance can turn a varying high-offset column into a false constant.
Pre-cast moments describe different values from those used by training.
Process-exit-only cleanup can accumulate one multi-gigabyte file per sweep
point.

### Acceptance evidence

- A production-width synthetic grid under a tiny budget proves column-thinned,
  row-chunked reads and a disk-backed final result.
- Values and means equal an independent stored-float32 known answer.
- Whole-selection and mean-before-cast mutations fail.
- High-offset uneven-chunk controls agree with NumPy population statistics.
- A fixture straddling the relative pin boundary distinguishes pre-cast from
  stored-payload moments categorically.
- Raw/base sizes `N-1`, `N`, and `N+1` cover exact row-count equality.
- Real loaders after the law transform preserve seeded order in resident and
  disk-backed regimes; an `arange` mutation fails.
- Restaging removes the first temporary file, a multi-point sweep holds at
  most one train file, failure leaves no partial file, and resident mode makes
  no temporary file.
- The sweep failure leg executes the real sweep function rather than manually
  imitating cleanup.

Optional production-scale polynomial chaos expansion (PCE) fitting is outside
this bounded-staging claim until a streamed or randomized low-rank fit and
accuracy contract exists.

## Data controls

### Rule

One pure validator runs before staging:

- `split_seed`: required native non-Boolean integer in the documented range;
- `ram_frac`: finite native non-Boolean real in `[0,1]`;
- every active cut bound: finite native non-Boolean real;
- paired lower and upper bounds: `lower < upper`;
- one-row text tables: normalized to exact two-dimensional form with validated
  column count.

No coercion is allowed. The train and validation files are configured
separately, and the library does not prove they hold different rows: the
loaders state explicitly that they do not test physical-row disjointness.
Naming two genuinely disjoint files is the user's responsibility, exactly
like every other scientific input choice.

### Acceptance evidence

Tests cover malformed seed and memory values, NaN/Inf/Boolean cut bounds,
reversed bounds, one-row validation, identical paths, symlink and hardlink
aliases, separately named duplicate payloads, partial overlap, and a valid
disjoint pair.

## Input validation before output

### Rule

Generator inputs are validated before any output file is created:

- `train_args.ord` must be one permutation of Cobaya's sampled parameter
  set (same names, same count);
- the family `train_args` keys are validated by each driver before the model
  requirements are registered (grid shapes and ordering, multipole range,
  `extrap_kmax >= max(k)`, the Syren-base switch a native Boolean);
- `--boundary`, `--maxcorr`, `--freqchk`, `--nparams`, and `--temp` have
  explicit legal ranges, and the command line parses strictly: an unknown or
  misspelled flag is a usage error, never silently ignored (the `cli-strict`
  gate);
- Cobaya's raw confidence-one prior endpoints may be infinite; every resolved
  bound used for sampling or written to an output file is finite and
  increasing, checked after the temperature stretch and boundary trim.

Optional `latex` is presentation metadata: when absent the parameter name is
the GetDist label, and its absence may not abort an otherwise valid run.

A Gaussian-mode unique-row shortfall (fewer unique MCMC rows than requested)
warns and continues with the smaller table; the row count is visible in every
output file.

### Acceptance evidence

- A wrong, duplicated, or missing `ord` name refuses before sampling.
- Invalid flag values and unknown flags refuse before output creation.
- The label fallback is proved by the CPU gate for a parameter with no
  `latex` entry.

## Data paths and axis identity

### Rule

`emulator/cocoa.py` resolves every file-valued key in the training config's
`data` block against the project `chains/` folder: the flat keys (dumps,
parameter files, covariance, failure masks) and the family sub-block keys
(`grid.z_file`, `grid2d.z_file` / `k_file` / `train_base` / `val_base`,
`cmb.covariance`). Absolute paths pass through unchanged. The YAML therefore
carries bare filenames and a driver may run from any working directory.

Coordinate axes are read from the generator's sidecar files, never re-declared
in a YAML. Each config names one axis file per coordinate, shared by the
training and validation payloads it describes; a payload trained against the
wrong axis file is a user configuration error the axis-shape checks at
staging and geometry construction catch when the widths disagree.

### Acceptance evidence

- Shipped family configs resolve from a working directory different from the
  chain directory.
- Absolute paths stay unchanged; a missing file fails naming its resolved
  path.
- A grid or multipole width that disagrees with the payload refuses before
  training.

## Physical parameter cuts

### Rule

Physical cuts live under `data.param_cuts`. The whitelist is:

- required: `omegabh2_hi` when the block is active;
- optional: `omegabh2_lo`, `omegam2h2_lo`, `omegam2h2_hi`,
  `omegamh2_lo`, `omegamh2_hi`, `omegamh2ns_lo`, `omegamh2ns_hi`.

Legacy flat keys and `omegabh2_cut` refuse with migration guidance. Each
window is a row in one quantity table containing name, label, required input
columns, formula, lower bound, and upper bound. In these formulas, `H0` is the
Hubble constant measured in kilometers per second per megaparsec; `omegab` is
the baryon density fraction; `omegam` is the total matter density fraction;
and `ns` is the scalar spectral index. The strict formulas are:

```text
omegabh2   = omegab * (H0/100)^2
omegam2h2  = (omegam * H0/100)^2
omegamh2   = omegam * (H0/100)^2
omegamh2ns = omegamh2 * ns
```

Active windows intersect. The banner names each formula and reports the
independently reproducible kept count. If the pool is too small, the error
reports kept and requested rows and explains the remedy. Cutting sparse volume
densifies the retained region because `n_train` is drawn after the cut.

### Acceptance evidence

- No-cut pool size equals the full table.
- Each individual formula and each stacked intersection has an independent
  known-answer survivor set.
- Pool sizing and staging select the same named rows in the same seeded order.
- A banner-only mutation with wrong rows fails.

## Triangle-plot cut shading

### Rule

Each active physical window shades only panels whose coordinates determine the
window. Every cut artist uses `_CUT_GREY = (0.55, 0.55, 0.55, 0.30)` at
z-order zero. Multiple windows compose by artist superposition. Plotting
formulas cite the shared cut-table helper. No fuzzy projection appears on a
panel that cannot determine the window.

## Stable evidence anchors

<a id="param-window-cuts-evidence"></a>
### Parameter-window driver evidence

The gate reads the resolved cut config and manifest-declared inputs, launches
the training driver, and records ordinary artifact and stream output. Its
three claims remain narrow.

<a id="param-window-cuts-driver-exit-zero"></a>
`param-window-cuts.driver-exit-zero` requires the driver subprocess to exit
zero.

<a id="param-window-cuts-cut-count-banner-present"></a>
`param-window-cuts.cut-count-banner-present` requires a line matching
`used N of P cut rows`. Presence alone does not prove an independent count.

<a id="param-window-cuts-init-probes-inspection"></a>
`param-window-cuts.init-probes-inspection` requires a real A/B comparison.
A manual inspection instruction does not satisfy this acceptance check.

<a id="triangle-shading-evidence"></a>
### Triangle-shading evidence

The synthetic plotting check maps each axes object to exact parameter
coordinates, traces every z-order-zero cut mask, and compares the complete
artist tuple set with an independent formula table.

<a id="triangle-shading-figure-produced"></a>
`triangle-shading.figure-produced` requires the helper to return a figure for
recognized parameter names.

<a id="triangle-shading-panel-window-set-exact"></a>
`triangle-shading.panel-window-set-exact` requires exact panel ownership,
window identity, mask, and artist count. Moving an artist to a wrong panel
must fail while global counts stay unchanged.

<a id="triangle-shading-all-cut-artists-use-shared-gray"></a>
`triangle-shading.all-cut-artists-use-shared-gray` requires every collection
and patch on the cut layer to match `_CUT_GREY` exactly.

<a id="triangle-shading-omegamh2-marginal-bands-exact"></a>
`triangle-shading.omegamh2-marginal-bands-exact` requires exactly two excluded
interval patches on the `omegamh2` diagonal, none elsewhere, with exact
endpoints.

## Refactor boundaries

The seven repeated family storage operations may move into one ordinary-loop
multi-array store in `generator_core.py`. Family physics and sidecar creation
stay in family drivers. Acceptance is byte identity across fresh, checkpoint,
and append paths. Consolidation follows correctness work on the same files so
the refactor does not multiply review churn.

The staging comments are part of the contract. A first-time reader must be able
to label every coordinate global or local and every access eager, lazy, view,
or copy without reconstructing old gate history.

## Claims that must remain explicit

- The failure mask and the scientific record are trusted as named: the
  library validates their form, not their pairing with a particular dump.
  The user owns the choice of which files belong together.
- Append reproducibility comes from deriving the stream from the seed plus
  the existing row count; a fresh `N` run followed by an append of `M` rows
  is replayable, but it is not row-identical to one fresh `N+M` run.
- Bounded grid2d staging does not certify production-sized PCE fitting.
- Temporary-file recovery does not promise survival of `SIGKILL`, the
  operating-system signal that stops a process without cleanup; an
  out-of-memory (OOM) termination; or storage-controller failure beyond the
  documented sync boundary.
- A valid scientific gate compares values with an independent known answer;
  a process exit, banner, shape, or previous log alone is not that evidence.
