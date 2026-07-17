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

A **publication transaction** turns one complete draft dataset into the
dataset visible to readers without exposing a partial copy. A **slot** is the
stable destination assigned to one append-stable request identity. A
**generation** is one complete version stored in that slot; a **draft**
remains writable, while a **sealed generation** no longer changes. A
**locator** connects the familiar parameter-chain filename in a training YAML
to its stable slot and request. It never names a particular generation. A
file **descriptor** is the operating system's handle for one open file.

A **digest** is a fixed-size fingerprint calculated from exact bytes. A
**manifest** is the complete list of a generation's members together with
their paths, lengths, and digests. The **active record** is
the small `active.json` file that names the sealed generation readers must
open. Replacing that record is the **active-pointer switch**.

A **random-engine policy** names every random-number algorithm and the saved
state needed to continue its sequence exactly. It is different from a seed,
which chooses only the initial state.

A **probe** names the requested physical output, such as background or CMB. A
family **variant** selects one documented representation, such as native or
Syren-base matter power. A **registry** is a fixed mapping of accepted names
to their owners. A **canonical projection** is the documented subset of
resolved settings, written in one stable order and encoding before its digest
is calculated.

Four identities carry a dataset into a saved emulator. **Request identity**
describes the intended scientific setup and stays stable across a valid
append. **Generation identity** binds that request to the exact sealed manifest
and every semantic member from one generator run. **Staged-selection identity**
binds both source generation identities, parameter rows, covariance, fixed
facts, axes, payloads, cuts, split rule, and final training and validation row
order.
**Artifact identity** then binds that staged selection to the resolved model,
training recipe, composition, and saved weights. The artifact note owns the
last identity. A matching filename or array shape proves none of them.

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
| Logical filename and generation resolution | `emulator/cocoa.py` |
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

<a id="cmb-covariance-publication-transactional-output"></a>
### CMB covariance publication

#### Rule

The CMB covariance archive is a generated scientific dataset, not an emulator
artifact. `compute_cmb_covariance.py` must refuse when any path entry already
owns the requested final name. This includes a regular file, a directory, a
symlink to an existing target, and a dangling symlink whose target is missing.
The command performs this check before it reads the YAML configuration or asks
CAMB to calculate a spectrum, so a rerun cannot spend computation or alter an
existing destination.

Publication uses one hidden, uniquely named staging file in the final file's
directory. After writing the zipped NumPy archive (`.npz`), the producer
synchronizes the staging file with `fsync`. It then opens that staging file
with `numpy.load(..., allow_pickle=False)` and reads every member. The member
names, dtypes, shapes, and values must exactly match the arrays supplied for
publication; missing, additional, reordered-by-substitution, or changed
members refuse publication.

Only a validated staging file may claim the final name. The producer uses an
atomic create-if-absent operation, so a file or link created by another
process after the early destination check wins unchanged. It never uses an
overwriting replacement for this final step. After the final name is created,
the producer synchronizes the containing directory so the new directory entry
reaches the documented durability boundary. It removes its staging file after
success and after every handled failure, including a write fault, a file-sync
fault, invalid readback, a late competing destination, or a directory-sync
fault.

#### Why

A covariance calculation can take substantial time, and downstream code
treats the final filename as a complete scientific result. Same-directory
staging and non-overwriting final-name creation ensure that readers see either
no result or one fully checked archive. They never see a partial archive, and
a concurrent producer never loses the result that reached the final name
first.

#### Acceptance evidence

The gate claim `cmb-covariance-publication.transactional-output` receives GO
only when all of the following checks pass:

- a successful publication reads back the exact member-name set and the exact
  dtype, shape, and value of every member with `allow_pickle=False`;
- an existing file and a dangling-link destination both refuse before YAML
  parsing, CAMB evaluation, or staging-file creation;
- faults at archive writing, staging-file synchronization, exact readback,
  final-name creation, and directory synchronization leave no trusted partial
  result and no staging file;
- a destination created after readback remains byte-for-byte unchanged, while
  the losing staging file is removed; and
- mutations that permit pickle loading, skip one member comparison, overwrite
  the final name, omit either synchronization step, or retain a staging file
  make the claim fail.

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

## Sampling and request or generation identity

### Rule

Fresh tempered sampling uses the resolved parameter covariance and an owned
random number generator. Fresh uniform sampling uses the resolved legal
interval after the boundary-interior policy has moved each endpoint one
representable value toward the interval interior. No boundary policy may
depend on distance from zero.

Temperature, maximum-correlation value and applicability, boundary policy,
requested and resolved support, sampling algorithm, seed, random-engine
policy, ordered parameter names, family, mode, scientific settings, and the
target-producing physics implementations belong to request identity.
Coincident numeric bounds do not erase the requested policy.

The current sealed generation does not save enough changing random-engine,
sampler, walker, log-probability, and unique-row-selection state to continue
the sequence exactly. Append therefore authenticates the active generation
and then refuses without drawing or publishing a row. Before exact append may
be enabled, that complete continuation state must become authenticated
checkpoint state. Reinitializing from the seed is not continuation.

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
- Current append intent authenticates the active generation and then refuses
  without changing it.
- A future exact-append implementation must prove that fresh `N` followed by
  append `M` produces the same canonical parameter rows and order as one fresh
  `N+M` run.
- A future append implementation must refuse missing, stale, or corrupt
  continuation state without changing the prior generation.
- A mutation that restores endpoint multiplication or
  `default_rng(seed)` at append must fail.

<a id="generator-seed-owned-rng"></a>
### Owned random-number generation

#### Rule

A required native integer seed initializes an owned NumPy PCG64 generator for
parameter draws, Gaussian walker initialization, and unique-row selection.
The emcee sampler owns a separate NumPy `RandomState` using MT19937 for its
moves. Process-global `np.random` draws are not part of the generator contract.
The current published members do not contain the complete states needed for
continuation. Before append may be enabled, it must authenticate both random
states, walker coordinates, walker log probabilities, and unique-row-selection
state; a seed alone is insufficient.

#### Acceptance evidence

- Equal seed and equal request produce equal rows.
- A different seed changes the selected rows.
- Current append refuses after authenticating the active generation. A future
  exact append must equal a one-shot run.
- Worker count does not change the canonical parameter table.
- A scan and mutation check prove that no process-global random draw enters the
  generation path.

## Canonical parameter rows

### Rule

One canonical parameter representation is materialized before any science
evaluation. Every family producer receives rows bitwise equal to the rows the
training loader later recovers from the published table. Fresh serial and MPI
paths share that representation and row order. Resume copies the authenticated
representation into a private draft. Append has no accepted row-producing path
until complete continuation state is saved.

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
- Fresh serial and MPI paths agree on canonical rows, and resume preserves the
  authenticated rows exactly. Current append refuses before producing rows.
- Scripted MPI replies prove that a valid out-of-order result reaches only its
  assigned row, while a wrong row, inactive worker, duplicate reply, malformed
  reply, and false stop acknowledgement change no stored row.
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

The production generator, Cocoa configuration resolver, and staging path use
this transaction. The lower-level publisher owns immutable files and the
active switch; the sections below also state where the production bridge adds
request binding, worker coordination, reader pinning, and row selection.

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

Rank zero prepares a private draft before sampling. If draft preparation or
sampling refuses, it broadcasts that refusal before any worker enters the
data-vector calculation. On success, every retained memory map is flushed and
closed, and the Message Passing Interface (MPI) barrier completes before rank
zero publishes. The barrier waits until every parallel process reaches the
same point.

The publisher opens and fingerprints every source before copying the first
one and keeps each file descriptor, the operating system's open-file handle,
active. After the final copy, the publisher rechecks the inode, the filesystem
identity of the underlying file, together with size, time tokens, and census.
Published members receive new sealed inodes. Resume copies an authenticated
active generation into a new mutable draft. Append authenticates the active
generation and then refuses until complete continuation state is available.
Neither path creates a hardlink or another writable name for a published
member.

#### Acceptance evidence

A source mutation at any acquisition or copy boundary refuses. Retained source
descriptors cannot change the sealed generation.

<a id="dataset-publication-copy-on-write-continuation"></a>
### Copy-on-write continuation

#### Rule

`begin_dataset_continuation` authenticates the requested active generation
before creating a new draft. It opens every published member before copying
the first member and keeps those file handles open until the complete copy has
been checked. Each copy has the authenticated size and SHA-256, a different
inode from its source, mode `0600`, and one filename. The draft and its member
directories have mode `0700` and exactly the declared files and directories.
Copied files and the complete draft directory tree are synchronized before
the function returns.

The source member handles and their named paths are rechecked after the last
copy. The manifest bytes, manifest SHA-256, read-only directory modes, and
complete source census are read and checked again at that point. The manifest
is authenticated by content; the operation does not claim that the manifest's
inode stays unchanged throughout the copy.

The returned `ContinuationDraft` retains the original
`ActiveGeneration.active_sha256`. If another writer selects generation B while
a continuation copies generation A, the A copy may finish, but a later
publication with A's saved active-record SHA-256 must refuse and leave B
active. That later publication refusal keeps the completed A draft for
inspection or retry.

A refusal while `begin_dataset_continuation` is preparing the copy asks for
best-effort removal of only the new draft. If that cleanup fails, the partial
draft may remain in the work folder. Preparation and cleanup do not change the
active record, the published source, or another draft in the same work folder.

#### Why

A hardlink or other writable alias would let continuation work change a
published dataset. Refreshing the saved active-record SHA-256 after a competing
writer succeeds would let stale work replace that writer's result. Separate
files and the original saved value prevent both failures.

#### Implementation boundary

This helper prepares a safe mutable copy. Generator resume uses it and keeps
the returned active-record digest for compare-and-swap publication. Exact
append, recovery of a first interrupted unpublished draft, saved random state,
and old-generation removal remain separate functionality. Cocoa reader
pinning and MPI publication order have their own production checks.

#### Acceptance evidence

- Request identity and the complete member map refuse before a draft exists.
- Nested members copy with identical bytes but different inodes and private
  writable modes.
- Writable, replaced, linked, changed, missing, extra, or wrongly sized source
  and draft entries refuse.
- Instrumentation proves member-file synchronization and final synchronization
  of every draft directory plus the work folder.
- Copy, validation, and file-close failures ask for best-effort removal of only
  the new draft; a cleanup failure may leave that draft for inspection.
- A concurrent A-to-B switch returns A's original saved value, and a later
  stale publication refuses while B remains active and keeps the completed A
  draft.

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
canonical parsed configuration digest, append-stable scientific-contract
digest, and ordered semantic or content identifiers for every target-producing
physics formula.
Immutable registries bind probe to family and generator and sampling mode to
algorithm and random-engine policy.

The scientific-contract digest hashes a versioned, resource-bounded canonical
projection of resolved facts. A generation-specific dataset or chain digest is
not part of the append-invariant request. Run controls and append row count are
also not scientific identity. The current member map contains no complete
continuation-state member. A future exact-append implementation must add and
authenticate that state without moving it into request identity.

#### Acceptance evidence

Requests with missing or unknown fields, Boolean seeds, wrong family or
algorithm, reordered parameters, omitted temperature, changed scientific
facts, or excessive canonical structures refuse. Current append intent keeps
the same invariant request identity, authenticates the active generation, and
then refuses without changing its chain digest. A future successful append
must keep that request identity while changing generation-specific content.

<a id="dataset-request-contract-family-members"></a>
### Family member map

#### Rule

Every chain-only generation has five members: chain, parameter schema,
covariance, ranges, and facts. Full generation adds the failure mask and:

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

<a id="failed-row-staging-and-selection-identity"></a>
### Failed rows and exact staged selection

#### Rule

A full data-vector source resolved by Cocoa requires the authenticated
failure-mask member from its pinned generation. The mask contains exactly one
ASCII `0` or `1` line for every parameter and payload row. A missing mask,
another token, or a different row count refuses. Chain-only scalar data has no
data-vector failure mask. A caller that invokes staging directly supplies its
own mask path; staging validates the tokens and row count but does not prove
that file came from a sealed generation. That direct caller owns the source
trust decision.

The generator may keep a private draft containing failed placeholder rows for
diagnosis, but a mask containing any `1` refuses publication. Staging removes
failed rows before physical cuts, seeded selection, and pool-size reporting.
It cannot satisfy a requested training size with a failed row.

Cocoa resolves training and validation once each and saves both generation
pins. After staging, the saved source pin adds the original source row count,
split seed, physical cuts, selected count, and an order-sensitive SHA-256 of
the exact disk rows addressed by the loader. For a compact resident source,
that order is `dump_rows[idx]`. For a disk-backed source, `idx` already holds
global disk rows. The recorder derives the applicable order from the staged
array sizes and refuses unless it equals `selected_rows` exactly.

#### Acceptance evidence

- A four-row source with mask `0, 1, 0, 1` can stage only rows zero and two.
- Missing, short, malformed, and failed-row masks refuse before training.
- Resident, disk-backed, scalar, and all-row selections save the disk order
  that the loader actually addresses.
- Equal counts with a different permutation refuse before the fingerprint is
  saved.
- Both saved configuration records retain the accepted generation and row
  selection identity.

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

<a id="generator-ingress"></a>
<a id="generator-ingress-valid-before-output"></a>
## Generator ingress

### Rule

`train_args.ord` is one nonempty list of unique native strings. Each name is one
visible token without whitespace or control characters. Cardinality,
membership, and order are checked against Cobaya's sampled parameter set, with
missing, extra, and duplicate names reported separately. The covariance
filename is one direct child of the YAML folder. Covariance headers are
nonempty and unique; matrices are finite, two-dimensional, square, positive on
the diagonal, and aligned before subsetting. Opposite entries may differ only
by floating-point roundoff and are then averaged to exact symmetry. A
covariance header may be a superset of the requested parameters, but its
complete ordered name list must match the complete matrix dimension before the
requested submatrix is selected. `fiducial` is a mapping from sampled parameter
names to finite non-Boolean numbers.

Family grid counts and multipoles are native non-Boolean integers. Switches are
native Booleans. Grid edges and extrapolation limits are finite. Unknown family
keys refuse. `extrap_kmax >= max(k)` after validation. Priors, covariance,
Cholesky factors, triangular matrices `L` satisfying `L @ L.T = covariance`,
inverses, modeled columns, and metadata are finite. A
Markov-chain Monte Carlo (MCMC) unique-row shortfall at the saved `float32`
parameter precision refuses rather than publishing a smaller dataset. Unknown
command-line arguments refuse.

Cobaya's raw confidence-one prior endpoints may be infinite when a prior is
unbounded. Those raw endpoints are permitted only as inputs to interval
resolution. Every resolved bound used for sampling or written to a published
file is finite and increasing. A finite raw endpoint that overflows the
generator's parameter dtype refuses rather than being mistaken for an open
endpoint.

Common and family-specific `train_args`, parameter order, covariance,
fiducials, and resolved bounds are validated before output creation. MCMC row
cardinality is validated after sampling but before output creation. Refusal
for one of these input rules, including a unique-row shortfall, must not create
`chains/`, an output draft, or a partial public dataset.

Optional `latex` is presentation metadata. When absent, `None`, or blank, the
parameter name is the GetDist label. Its absence may not abort an otherwise
valid run. A label may contain ordinary spaces and LaTeX commands, but it may
not contain a line break, tab, NUL byte, or another control character that can
split a saved table.

### Acceptance evidence

Unique reorder controls preserve index maps. Duplicate, missing, extra,
wrong-nesting, non-string, whitespace-bearing names, escaping covariance paths,
header/matrix mismatch, nonfinite covariance or fiducial, unsafe display-label
controls, lossy integer/Boolean coercion, unknown key, and unknown flag cases
all refuse before sampling or output mutation. A unique-row shortfall refuses
after MCMC selection and before output mutation. The CPU gate proves the label
fallback and that the production writer uses the label prepared during setup.
When a release changes the Cobaya or GetDist boundary, a separate workstation
acceptance run should also publish a real Cobaya parameter with no `latex` and
read the resulting sidecar through the installed GetDist.

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
- Production YAML filenames are logical locator keys, not mutable flat-file
  paths. Missing locators require explicit regeneration or migration; there is
  no legacy flat-file fallback.
- A first interrupted unpublished draft cannot yet be resumed. Exact append
  also refuses because NumPy, sampler, walker, log-probability, and unique-row
  selection state are not yet persisted. Both limitations fail closed and do
  not change an earlier active generation.
- Bounded grid2d staging does not certify production-sized PCE fitting.
- Temporary-file recovery does not promise survival of `SIGKILL`, the
  operating-system signal that stops a process without cleanup; an
  out-of-memory (OOM) termination; or storage-controller failure beyond the
  documented sync boundary.
- A valid scientific gate compares values with an independent known answer;
  a process exit, banner, shape, or previous log alone is not that evidence.
