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
axes, or other facts about a larger data file. A **gate** is a registered
acceptance command.

A **publication transaction** turns one complete draft dataset into the
dataset visible to readers without exposing a partial copy. A **slot** is the
stable destination assigned to one dataset identity. A **generation** is one
complete version stored in that slot; a **draft** remains writable, while a
**sealed generation** no longer changes. A file **descriptor** is the
operating system's handle for one open file.

A **digest** is a fixed-size fingerprint calculated from exact bytes. A
**manifest** is the complete list of a generation's members together with
their paths, lengths, and digests. The **active record** is
the small `active.json` file that names the sealed generation readers must
open. Replacing that record is the **active-pointer switch**.

A **probe** names the requested physical output, such as background or CMB. A
family **variant** selects one documented representation, such as native or
Syren-base matter power. A **registry** is a fixed mapping of accepted names
to their owners. A **canonical projection** is the documented subset of
resolved settings, written in one stable order and encoding before its digest
is calculated.

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
| Dataset publication transaction | `compute_data_vectors/dataset_publication.py` |
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

The shared core prevents checkpoint and publication rules from drifting among
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

## Sampling and dataset identity

### Rule

Tempered sampling uses the resolved parameter covariance and a recorded random
number generator. Uniform sampling uses the resolved legal interval after the
boundary-interior policy has moved each endpoint one representable value toward
the interval interior. No boundary policy may depend on distance from zero.

Temperature, maximum-correlation value and applicability, boundary policy,
requested and resolved support, sampling algorithm, seed, complete
random-generator state, ordered parameter names, family, mode, and all
scientific settings are dataset identity. Coincident numeric bounds do not
erase the requested policy.

The complete generator-owned random state is checkpoint state. Append restores
that state before drawing another row. Reinitializing from the seed is not
continuation.

### Why

Uniform support changes when infinite prior endpoints are resolved with a
temperature. A filename that records only `unifs` cannot distinguish those
scientific supports. Seed-only append restarts the random stream and can
duplicate the original dataset.

### Acceptance evidence

- Same-width intervals translated along the number line retain the same
  fractional interior width.
- Narrow offset intervals either have a representable interior or refuse before
  output mutation.
- Two temperatures produce distinct request identities even if hard bounds make
  their effective numeric supports equal.
- Fresh `N` followed by append `M` produces the same canonical parameter rows
  and order as one fresh `N+M` run.
- Missing, stale, or corrupt random state refuses append without changing the
  prior generation.
- A mutation that restores endpoint multiplication or
  `default_rng(seed)` at append must fail.

<a id="generator-seed-owned-rng"></a>
### Owned random-number generation

#### Rule

A required native integer seed initializes an owned NumPy generator. The same
owned generator supplies uniform draws, Gaussian walker initialization, sampler
moves, and thinning or subselection. Process-global `np.random` draws are not
part of the generator contract. The complete state, not merely the seed, is
recorded for continuation.

#### Acceptance evidence

- Equal seed and equal request produce equal rows.
- A different seed changes the selected rows.
- Append continuation equals a one-shot run.
- Worker count does not change the canonical parameter table.
- A scan and mutation check prove that no process-global random draw enters the
  generation path.

## Canonical parameter rows

### Rule

One canonical parameter representation is materialized before any science
evaluation. Every family producer receives rows bitwise equal to the rows the
training loader later recovers from the published table. Fresh, resume, append,
serial, and MPI paths share that representation and row order.

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
- Fresh, append, resume, serial, and MPI paths agree on canonical rows.
- GetDist selects the higher-posterior row in a two-row known answer.
- The reserved-column sign mutation selects the wrong row and must fail.
- Float32-distinct range-bound pairs serialize and read back as distinct,
  ordered bounds containing all canonical rows.

## Dataset readiness

### Rule

A row is successful only when all of these statements are true:

1. The lifecycle accepted that exact cosmology.
2. The complete family payload has the exact declared key set and shape.
3. Validation runs after conversion to the exact storage dtype.
4. Every stored value is finite.
5. Family-specific domain rules hold.
6. The row was written once and the accepted cast payload can be read back.
7. Only then may the failure flag be cleared.

Failure metadata belongs to the same dataset generation as the parameter
table, payloads, axes, and facts. Missing, stale, forged, or wrong-length
failure metadata makes the dataset unreadable.

A production-ready dataset contains the exact requested number of successful
rows. A loader may reject failed rows or explicitly exclude them, but an
exclusion policy still has to deliver the requested successful-row count.
Failed proposals may remain in a retry checkpoint; they are not training data.

`boundary` is a finite native non-Boolean real with
`0 < boundary <= 1`. Invalid values refuse before any output path is opened.

### Why

A provider failure can leave a zero-filled row while the program exits zero.
Finiteness before a float32 cast misses overflow created by that cast. A
broadcast-compatible shape can fill a row without representing the declared
family structure. Structural checkpoint checks alone can also republish a
not-a-number value (`NaN`) or infinity (`Inf`) under a success bit.

### Owner

- lifecycle acceptance: the shared lifecycle helper;
- stored-payload predicate: one shared family-aware validator;
- resume revalidation: every family checkpoint loader;
- final readiness: generator publication and staging entry points.

### Acceptance evidence

- Finite control rows publish.
- NaN, positive or negative Inf, and finite-float64-to-float32 overflow fail.
- A second-row scalar or length-one payload cannot broadcast into a full row.
- Missing or extra family keys fail.
- Signed CMB TE remains legal while family auto-spectrum rules remain active.
- Serial and MPI result handlers invoke the same validator.
- A successful flag paired with an invalid resumed payload refuses before a
  “loaded” report and leaves all files unchanged.
- A failed-row zero placeholder remains legal.
- Boundary controls cover zero, negative, above one, NaN, Boolean, one, and a
  valid interior value.
- Restoring unconditional success or removing the staging-side readiness check
  makes the gate fail.

## Native-output and lifecycle acceptance

### Rule

Native-output capture is supplementary diagnostic evidence. The scientific
acceptance fact is a finite accepted Cobaya `LogPosterior.logpost`, checked
before the first getter call. Terminal keyword scans never decide scientific
success.

Python streams are flushed before file-descriptor redirection. Every supported
writer is flushed after the theory call and before reading captured text. If a
native writer cannot be synchronized reliably, the solver must be isolated
behind a process boundary. The other valid choice is a supported application
programming interface (API) for status or exceptions. An API here means the
documented calls that report solver state. Capture cleanup and descriptor
restoration remain exception-safe.

### Why

Buffered Python or C output can appear in the next sample's capture. A rejected
Cobaya point normally returns `-inf`; it need not raise or print a keyword.
The provider can still contain a finite array from an earlier accepted point.

### Acceptance evidence

- Immediate `os.write`, buffered Python output, and a genuinely buffered native
  writer are captured in the correct sample.
- Text written before entry is not assigned to the sample.
- Exceptions restore both descriptors.
- A rejected result with a stale finite provider array performs zero getters
  and cannot write.
- An accepted lifecycle followed by a nonfinite payload reaches the payload
  validator and fails there.
- Generator and gate-side lifecycle calls use the same acceptance definition.

## Run-control state

<a id="generator-run-control-binary-state"></a>
### Binary controls

#### Rule

`loadchk`, `append`, and `chain` accept only native non-Boolean integers
`0` or `1`. The normalized run-control record is immutable. Legal
`(loadchk, append)` pairs are exactly `(0,0)` for fresh generation, `(1,0)`
for resume, and `(1,1)` for append. `chain` independently selects `full` or
`chain-only` mode.

#### Acceptance evidence

- Boolean and coerced numeric controls refuse.
- The three legal operation pairs reach their intended branches.
- Every other pair refuses before setup.

<a id="generator-run-control-append-requires-load"></a>
### Append requires a validated prior generation

#### Rule

`append=1, loadchk=0` is illegal. The error names both values and explains
that append extends a validated prior dataset. It never means fresh generation
at an existing path.

#### Acceptance evidence

A complete sentinel dataset keeps every byte and modification time after the
illegal command. Restoring independent flag handling must visibly damage the
sentinel and fail the gate.

<a id="generator-run-control-pre-mutation-refusal"></a>
### Validation precedes setup

#### Rule

The constructor assigns the direct run-control validator before environment
lookup, configuration reads, directory creation, or setup. Setup consumes the
normalized values. Final mode selection consumes the normalized record rather
than raw command-line attributes.

#### Acceptance evidence

Filesystem sentinels prove that invalid intent creates no setup event or file.
Legal full mode reaches sampling and family work. Legal chain-only mode reaches
sampling but no data-vector work. Statement-order, hidden-expression,
shadow-binding, raw-value, and setup-first mutations must fail.

## Requested checkpoint refusal

<a id="checkpoint-refusal-missing-member"></a>
### Complete member census

#### Rule

Fresh mode may start without a checkpoint. Resume and append require the exact
mode- and family-specific member census. Missing members are all named, and no
surviving member is changed.

#### Acceptance evidence

Each generic and family member is removed in turn for resume and append. Every
case refuses before parsing or writing, with unchanged bytes and timestamps.

<a id="checkpoint-refusal-corrupt-load"></a>
### Corrupt checkpoints remain errors

#### Rule

Checkpoint readback uses the shared producer-schema resolver. Bookkeeping,
sampled, and derived numeric cells must be finite. The exact producer
`.paramnames` sidecar owns sampled order and the final `chi2*` declaration.
Failure lines are literal `0` or `1` producer tokens. Similar numeric text,
shadow sidecars, control separators, and mislabeled derived columns refuse.

#### Acceptance evidence

Tests cover a resolver error, nonfinite values in every column role, wrong
sidecar root, wrong derived label, and malformed failure tokens. Three
physical line endings are accepted: line feed (LF), used on Unix; carriage
return followed by line feed (CRLF), used on Windows; and bare carriage return
(CR), used by older systems. Vertical tab, form feed, and record separators
are not reinterpreted as rows. Every refusal preserves the original cause and
all checkpoint bytes and times.

<a id="checkpoint-refusal-no-fresh-fallback"></a>
### No fresh fallback

#### Rule

Only normalized `fresh` may enter the fresh writer. A failed resume or append
may not become fresh generation. Valid resume reads without sampling.

#### Acceptance evidence

Sentinels at the first fresh operation stay untouched when requested loaders
raise. The exception chain retains the original failure. A fallback mutation
must touch the sentinel and fail.

<a id="checkpoint-refusal-family-geometry"></a>
### Family geometry at checkpoint readback

#### Rule

Background and MPS axes are one-dimensional, have the configured shape, and
equal the configured coordinates exactly. CMB members have widths consistent
with the configured inclusive multipole range until the persisted CMB axis is
the authoritative manifest member.

#### Acceptance evidence

Changed background coordinates, short MPS axes, and short CMB spectra refuse.
Valid controls pass. Removing the axis or width checks must fail.

## Full and chain-only isolation

<a id="generator-run-control-dataset-mode-isolation"></a>
### Rule

Chain-only parameter, failure, and data-vector stems carry `_chain_only`.
Scoping all stems prevents later code from borrowing an unsuffixed full-dataset
member. A requested chain-only resume or append requires exactly covariance,
parameter names, ranges, chain, and facts, then returns before any failure,
payload, or axis I/O. Full mode keeps the full family census.

### Why

A chain-only run at full-dataset stems can replace parameter rows while leaving
old payload rows and failure flags, creating a same-shaped but scientifically
false pairing.

### Acceptance evidence

- Full and chain-only stems are distinct.
- Chain-only resume and append read exactly five parameter-side members.
- Chain-only paths perform no family payload or axis I/O.
- Full mode retains the complete family census.
- Unscoped stems, borrowed members, and a bypassed or misplaced mode barrier
  make the focused gate fail.

## Immutable dataset publication

This foundation defines the publication transaction. Generator and consumer
integration must not claim atomic datasets until they both use it.

<a id="dataset-publication-slot-identity"></a>
### Slot identity

#### Rule

A slot identity binds portable parameter, payload, and failure basenames,
family, and explicit mode. Relocating the complete `chains/` directory keeps
the identity. Changing any identity axis changes it. Stems must remain distinct
on case-insensitive filesystems. Descriptor, manifest, and active records use
one sorted whitespace-free JavaScript Object Notation (JSON) encoding in
Unicode Transformation Format, 8-bit (UTF-8), with an LF terminator. JSON is
the key-and-value text format used for these records; UTF-8 defines how that
text is stored as bytes. Unknown or duplicate keys, nonfinite values,
unsupported types, and unbounded integers refuse.

#### Acceptance evidence

Relocation preserves the slot id; changing each identity field changes it.
Canonical-format and resource-bound mutations refuse.

<a id="dataset-publication-exact-census"></a>
### Exact census

#### Rule

The caller supplies a complete semantic role-to-relative-path map. Every role
and path occurs once. Observed files and directories equal the declaration.
Missing, extra, empty, traversing, linked, symlinked, special, renamed, or
writable published entries refuse. The manifest records path, length, and
SHA-256, the cryptographic digest of the exact file bytes. Readers provide the
expected identity and exact census.

#### Acceptance evidence

Neither a familiar digest at a wrong basename nor a valid subset of a larger
generation is accepted. Every unsafe entry class has a refusal leg.

<a id="dataset-publication-sealed-epoch"></a>
### Sealed epoch

#### Rule

All writers close and the Message Passing Interface (MPI) barrier completes
before publication. The barrier waits until every parallel process reaches
the same point. The publisher opens and fingerprints every source before
copying the first one and keeps each file descriptor, the operating system's
open-file handle, active. After the final copy, the publisher rechecks the
inode, the filesystem identity of the underlying file, together with size,
time tokens, and census. Published members receive new sealed inodes. Resume
and append copy an authenticated generation into a new mutable draft; they
never create a hardlink, a second filename for the same inode, and reopen a
published member through it.

#### Acceptance evidence

A source mutation at any acquisition or copy boundary refuses. Retained source
descriptors cannot change the sealed generation.

<a id="dataset-publication-atomic-switch"></a>
### Atomic active pointer

#### Rule

One per-slot advisory lock serializes compliant publishers. A publisher
compares the SHA-256 of the complete previously read active record. It installs
the sealed generation before replacing `active.json` last. A reader opens the
active record once and resolves paths below one named generation. Garbage
collection retains that generation for the entire consuming operation.

#### Acceptance evidence

A live reader sees complete generation A or complete generation B, never mixed
members. A stale active token refuses even when only the generation field
changed.

<a id="dataset-publication-durability-and-recovery"></a>
### Durability and recovery

#### Rule

Sealed members, manifest, installed tree, the temporary file that will become
`active.json`, the replacement of `active.json`, and the slot directory are
synchronized in that order. Here **synchronized** means that pending writes
are forced from memory toward durable storage. Regular files use the strongest
available platform synchronization, including Darwin `F_FULLFSYNC` when
available. Directory creation synchronizes both the child and parent on every
retry. A refusal before the active-pointer switch removes only temporary
publication material and preserves the source draft. Cleanup after a
successful switch cannot turn a committed publication into a reported
failure.

#### Acceptance evidence

Instrumentation proves a successful sync after final read-only mode for every
member and manifest, an installed-generation directory sync before pointer
replacement, and an immediate slot-directory sync afterward. Retry after a
parent-sync failure repeats both child and parent syncs. Fault injection shows
the old pointer before replacement and a complete new pointer afterward.

## Invariant dataset request

<a id="dataset-request-contract-identity"></a>
### Request identity

#### Rule

One strict request schema binds mode, generator, family, probe, variant,
sampling mode, temperature, boundary, maximum-correlation applicability,
algorithm, seed, random-engine policy, ordered float32 parameter names,
resolved configuration digest, and append-stable scientific-contract digest.
Immutable registries bind probe to family and generator and sampling mode to
algorithm and random-engine policy.

The scientific-contract digest hashes a versioned, resource-bounded canonical
projection of resolved facts. A generation-specific dataset or chain digest is
not part of the append-invariant request. Run controls and append row count are
also not scientific identity. Complete continuation state is a separate
authenticated generation member.

#### Acceptance evidence

Requests with missing or unknown fields, Boolean seeds, wrong family or
algorithm, reordered parameters, omitted temperature, changed scientific
facts, or excessive canonical structures refuse. Valid append requests keep
the same invariant identity while their generation-specific chain digest
changes.

<a id="dataset-request-contract-family-members"></a>
### Family member map

#### Rule

Every chain-only generation has five members: chain, parameter schema,
covariance, ranges, and facts. Full generation adds failure state and:

| Family | Additional members |
|---|---|
| CosmoLike | one vector payload |
| CMB | TT, TE, EE, PP, and exact integer multipole axis |
| Background grid | Hubble rate `H` and transverse comoving distance `D_M`, each with its redshift axis |
| Native grid2d | linear power, boost, redshift axis, wavenumber axis |
| Syren-base grid2d | native members plus both base arrays |

The two base members are all-or-none. Temporary files, locks, caches, manifest,
and pointer are not semantic members. CMB multipoles are never inferred from
width, covariance, filename, or configured range.

#### Acceptance evidence

Every family and variant has an exact-census control. Missing CMB axis,
one-sided Syren base, extra cache files, or chain-only borrowing fails.

<a id="dataset-request-contract-generator-member-binding"></a>
### Generator member binding

#### Rule

Each generator binds its route and progress-file names once, after it reads
the driver-specific settings and chooses the full or chain-only output stems.
The running class's defining Python filename supplies the generator name. The
immutable probe registry must agree with that filename, the output family,
and the family variant.

The parameter, data-vector, and failure stems are normalized to absolute
paths and must share one folder. Only their portable basenames enter the
immutable member census. Checkpoint preflight joins that saved folder to the
saved member names; it does not ask a family driver to rebuild a second list.
Chain-only mode therefore remains the same five common members for every
family.

For matter-power generation, `write_syren_base` must be a YAML Boolean. A
false value binds the native Grid2D variant; a true value binds the
Syren-base variant and both base members. Fixed scientific facts reuse the
saved family, variant, and generator. Changing a probe, filename, or base
switch later cannot reclassify an already bound run.

This early census is not the complete dataset request and is not a published
manifest. It deliberately contains no configuration digest, scientific
digest, random state, transaction state, or consumer pin.

#### Acceptance evidence

Both modes and every family or variant produce the same members as the full
request contract. The complete family checkpoint methods agree with the
bound full census. A wrong driver filename, mismatched folder, case-colliding
stem, missing module filename, invalid family variant, or non-Boolean base
switch refuses before a checkpoint file is opened. Independent source
mutations fail when they restore the old family list, copy the expected
driver instead of observing the running file, erase the matter-power
variant, lend full members to chain-only mode, or recompute fixed facts from
mutable state. Further mutations fail when they move the binding before
driver settings and scoped stems, or bypass the matter-power Boolean
validator at its real assignment. Family-parity mutations also fail when a
lensing driver adds an override or when CMB and background change their
source-owned member constants without the canonical census changing with
them.

<a id="dataset-request-contract-mutation-controls"></a>
### Load-bearing request controls

#### Rule

Mutation evidence must cover omission or corruption of temperature, parameter
order, boundary policy, seed type, scientific digest, CMB axis, Syren base
pairing, chain-only census, family/generator mapping, algorithm, path-collision
checks, canonical integer bounds, append-stable projection, and recursive
resource bounds.

#### Acceptance evidence

Every listed mutation must be observed by a distinct refusal. A test that
mutates two consumers together does not prove either consumer independently.

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

No coercion is allowed. Train and validation paths must not alias by realpath,
symlink, hardlink, duplicate payload, or row identity. A same-pool split is
unsupported and refuses. Any supported split must be one explicit
partition operation with an empty-intersection proof.

### Acceptance evidence

Tests cover malformed seed and memory values, NaN/Inf/Boolean cut bounds,
reversed bounds, one-row validation, identical paths, symlink and hardlink
aliases, separately named duplicate payloads, partial overlap, and a valid
disjoint pair.

## Generator ingress

### Rule

`train_args.ord` is one nonempty list of unique native strings. Cardinality,
membership, and order are checked against Cobaya's sampled parameter set, with
missing, extra, and duplicate names reported separately. Covariance headers are
nonempty and unique; matrices are finite, two-dimensional, square, and aligned
before subsetting. Fiducials are finite non-Boolean numbers.

Family grid counts and multipoles are native non-Boolean integers. Switches are
native Booleans. Grid edges and extrapolation limits are finite. Unknown family
keys refuse. `extrap_kmax >= max(k)` after validation. Priors, covariance,
Cholesky factors, triangular matrices `L` satisfying `L @ L.T = covariance`,
inverses, modeled columns, and metadata are finite. A
Markov-chain Monte Carlo (MCMC) unique-row shortfall refuses rather than
publishing a smaller dataset. Unknown command-line arguments refuse.

Optional `latex` is presentation metadata. When absent, the parameter name is
the GetDist label. Its absence may not abort an otherwise valid run.

### Acceptance evidence

Unique reorder controls preserve index maps. Duplicate, missing, extra, wrong-
nesting, non-string, header/matrix mismatch, nonfinite covariance or fiducial,
lossy integer/Boolean coercion, unknown key, unique-row shortfall, and unknown
flag cases all refuse before sampling or output mutation. A real-Cobaya
parameter without `latex` completes sidecar and covariance publication and is
read back by GetDist.

## Nested data paths and axis identity

### Rule

One dotted-path registry resolves every file-valued config leaf against its
documented project base. Absolute paths pass through. Errors name the full
dotted key. Resolved consumed paths or a documented portable-root form are
persisted consistently.

Train and validation payloads each declare their own axes. Exact train/val
axis equality is required before staging. The dataset manifest binds every raw
and base payload to axis bytes, parameter order, failure state, settings, and
generation id. CMB payloads carry their own exact integer multipole sidecar;
width is never axis identity.

### Acceptance evidence

- Shipped family configs resolve from a working directory different from the
  chain directory.
- Absolute paths stay unchanged; missing paths name their dotted key; an old
  cwd-relative decoy is not consulted.
- Shifted, reversed, permuted, duplicated, gapped, and same-width wrong axes
  refuse before geometry or training.
- Separately written but byte-identical axes pass.
- CMB missing or anonymous multipole identity refuses with migration guidance.

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

- A member census is not a cryptographic dataset manifest.
- A manifest authenticates bytes and identity; it does not replace payload
  validity, axis semantics, or lifecycle acceptance.
- Dataset-publication primitives do not make current flat generator paths
  atomic until generator and consumer integration uses them.
- Bounded grid2d staging does not certify production-sized PCE fitting.
- Temporary-file recovery does not promise survival of `SIGKILL`, the
  operating-system signal that stops a process without cleanup; an
  out-of-memory (OOM) termination; or storage-controller failure beyond the
  documented sync boundary.
- A valid scientific gate compares values with an independent known answer;
  a process exit, banner, shape, or previous log alone is not that evidence.
