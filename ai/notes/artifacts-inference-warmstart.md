# Artifacts, inference, adapters, and warm starts

This note is the durable specification for saved emulators, reconstruction,
public prediction, adapters to Cobaya, fine-tuning, and transfer learning.
Cobaya is the Bayesian inference program that requests predictions from the
saved emulator. This note is not a development diary. Every section states a
durable rule, its reason, the code that owns it, and the acceptance evidence
that distinguishes the rule from a plausible but scientifically wrong
implementation.

Primary code owners are `emulator/results.py`, `emulator/inference.py`,
`emulator/warmstart.py`, `emulator/losses/transfer.py`,
`emulator/geometries/`, and `cobaya_theory/`.

## Vocabulary used throughout this note

An **artifact** is one complete saved emulator result with a shared path root.
**Publication** moves a complete, checked result into the final saved path
that readers use. **Provenance** records the inputs, source, and resolved
settings that produced a result; provenance records origin but is not by
itself proof that the result is correct. A publication **transaction** treats
both artifact files as one update. An **atomic artifact pair** never lets a
reader accept one old member together with one new member.
The `<root>.emul` file is a PyTorch checkpoint: its `state_dict` maps stable
names to model-parameter and registered-buffer tensors stored on the central
processing unit (CPU). The matching `<root>.h5` file uses Hierarchical Data
Format version 5 (HDF5) to store reconstruction metadata and scientific
arrays. A **schema** is the versioned list and meaning of those saved fields.
SHA-256 is the cryptographic digest used to bind the exact checkpoint bytes to
the HDF5 record.

A **random-engine policy** names every random-number algorithm and the saved
state needed to continue its sequence exactly. It is different from a seed,
which chooses only the initial state.

YAML is the human-readable settings-file format used by training and Cobaya
configurations. NumPy is the array library used for saved scientific tables;
PyTorch provides tensors, trainable models, and accelerator execution.
CUDA is NVIDIA's accelerator-computing platform. A graphics processing unit
(GPU) is the accelerator device used by CUDA training checks.

CoCoA is the surrounding cosmological-analysis installation. CosmoLike is the
likelihood calculation within that installation that produces and consumes
several emulator data-vector families.

The science-family abbreviations are:

- **CMB**, cosmic microwave background spectra such as temperature-temperature
  (`TT`), temperature-polarization (`TE`), and polarization-polarization
  (`EE`);
- **BAOSN**, background expansion outputs used with baryon acoustic
  oscillation and supernova data, including the Hubble rate and transverse
  comoving distance `D_M`;
- **MPS**, the matter-power spectrum, where `pklin` denotes linear power and
  `boost` denotes the nonlinear-to-linear power ratio;
- **IA**, intrinsic alignment of galaxy shapes, including the nonlinear
  alignment (`NLA`) and tidal-alignment/tidal-torquing (`TATT`) models;
- **PCE**, polynomial chaos expansion, and **NPCE**, a neural model that
  corrects a frozen PCE base;
- `xi`, the cosmic-shear correlation function, and `gammat`, the tangential
  shear used for galaxy-galaxy lensing.

`grid` means a quantity sampled along one physical coordinate. `grid2d` means
a surface sampled over redshift and wavenumber. **Syren** names the analytic
linear-power or nonlinear-boost baseline that some MPS artifacts correct.
`dv` means data vector, and `chi2` means the covariance-weighted squared error.
Whitened values have been rotated and scaled so the covariance is the identity;
physical values use the original scientific coordinates and units.

A **gate** is a named validation job whose required result is written before
it starts. A **leg** is one named
assertion in that gate, identified by an evidence identifier. A **child** is a
subprocess that runs gate checks; a **wrapper** is the harness code that starts
the child and any external commands. A child prints `##AID` records to bind
results to evidence identifiers. The **return code** (`rc`) is the child's
process result. A **control** is the valid case that must pass. A **mutation
arm** deliberately weakens one rule and must make its owning leg fail. A
**capability boundary** lists required software or hardware. A gate is
**capability-skipped** only when that requirement is absent; **unavailable**
means that a declared acceptance action did not run and therefore cannot count
as passing evidence.

The **driver root** is the project directory passed to the training program by
the gate configuration. The **driver file root** is the configured filename
stem for that training run. Together they determine generated paths such as
`<driver_root>/chains/gates_emul_evaluate.h5`.

## Two identities connect generation to inference

Identity answers “is this the same saved object?” Compatibility answers “may
these different objects be used together?” The library keeps two identities
separate:

1. **Staged-selection identity** records which source rows a training run
   staged and in what order: the source row count, split seed, physical
   cuts, selected count, and an order-sensitive digest of the exact disk
   rows, one record for training and one for validation, saved in the run's
   configuration. The scientific description of the dataset itself lives in
   the generator's `.facts.yaml` record beside the chain.
2. **Artifact identity** binds the saved model pair, resolved model and
   training recipes, output decoder and loss composition, staged-selection
   identity, composition mode, and any source artifact or analytic base.

An artifact records the exact staged selection for provenance. Public
inference does not need the original training files, but it must prove that
the requested family, product, parameter order, fixed scientific facts,
physical support, decoder, and analytic-base implementation are compatible
with the artifact. Matching shapes or filenames are never compatibility
evidence.

## Save and rebuild rule

Never reconstruct an artifact from code defaults. Defaults can change while a
saved model remains in use.

- **Write side:** save every value the run consumed, with all defaults
  materialized at save time.
- **Read side:** reconstruct only from saved values. A missing required key is
  a loud error that names the key and the migration or re-save action. The
  loader never substitutes a runtime code default.
- **Display side:** human-readable summaries render saved values but do not
  become a second source of truth.

## Schema 3 artifact contract

- One emulator uses one path root and two files. Here `<root>` is the shared
  filename stem. `<root>.emul` is a PyTorch checkpoint containing a
  `state_dict`, the mapping from parameter or registered-buffer names to
  tensors. Every checkpoint tensor is moved to CPU host memory, and the
  compiled-model `_orig_mod.` prefix is removed.
  `<root>.h5` is a Hierarchical Data Format (HDF5) file containing the
  scientific and reconstruction record.
- The HDF5 record contains the raw `config_yaml` and `train_args_yaml` as
  provenance; `config_resolved_yaml`, which includes the resolved training and
  data blocks; the staged-selection identity for both training and validation;
  and the `model_recipe` dataset. The model recipe names one supported class
  and records every constructor value used by that class. The file separately
  stores the parameter and data-vector geometry state, analytic-law name, and
  composition mode; polynomial-chaos-expansion (`pce`) or
  transfer-base state when applicable; training histories; and root attributes
  for `schema_version=3`, the Git commit, the Torch version, `rescale`, and
  family-specific facts.
- **Production checks before expensive work.** `from_yaml` and `from_config`
  first report errors that can be found in the configuration itself. They then
  read the training and validation scientific-record sidecars and compare each
  record's ordered sampled names with the covariance header. Both records must
  pass before the run chooses an accelerator, opens a warm-start or transfer
  artifact, or constructs the experiment. The approved text is retained and
  passed into staging; staging does not reopen the path and silently adopt
  different text. A low-level direct loader still performs the same record and
  name checks before it opens a large data-vector file.
- **One current writer format.** `save_emulator` requires the producer's exact
  scientific-record text plus plain mappings for the resolved training and
  model instructions. It checks that the recipe has every required field and
  uses known class, activation, normalization, and compile names. The trusted
  constructors and factories check their own numerical values later. The
  structural checks run before
  `model.state_dict`, temporary-file creation, pair markers, or replacement of
  an existing artifact. Every successful new save writes schema 3. There is no
  flag that writes schema 1, schema 2, or a file without a schema.
- **Legacy data are regenerated explicitly.** A missing `.facts.yaml` record
  is not permission for training to invent scientific metadata. The run stops
  with the generator's regeneration instruction.
- **A complete recipe is checked before imports.**
  `emulator/model_recipe.py` contains a closed description of all six
  supported model classes and their exact constructor fields. Its validator
  uses plain, non-executing Python values and imports no model, geometry,
  activation, normalization, or Torch implementation. A missing field, an
  unknown field, or an incomplete activation description therefore refuses
  before saved text can select a Python class. Numerical limits remain with
  the real constructor or factory that uses the value. The same structural
  validator covers an embedded transfer base. Explicit `head_act: null` means
  “inherit the trunk activation”; a missing `head_act` has no such meaning and
  refuses.
- **The saved recipe must describe the live object.** Each supported model
  constructor attaches the canonical recipe for the object it actually made.
  Before publication, `save_emulator` compares that live recipe with the
  claimed root recipe and, for transfer, with the claimed embedded-base
  recipe. A caller cannot save ordinary `ReLU` behavior under a registered
  gated-activation name, change the number of residual layers, or substitute a
  different class while retaining a plausible dictionary.
- **Saved geometry facts must fit both the recipe and one another.** The
  import-free check validates the class-specific fields and dimensions for
  parameter, log-parameter, amplitude-factor, data-vector, scalar, CMB, grid,
  and grid2d geometries. It checks such facts as ordered names, square bases,
  masks, kept indices, amplitude indices, and physical coordinate lengths.
  A self-consistent model recipe cannot make an inconsistent geometry safe.
- Every geometry group carries a `"cls"` attribute with the full module path.
  Rebuild resolves the stored string through `importlib` and calls that
  class's `from_state` method. A missing marker raises a `KeyError` that names
  the re-save action; rebuild never falls back to a base class. The file must
  identify the geometry type as well as store its numbers.
- `rebuild_emulator(path_root, device)` in `results.py` uses the HDF5 recipe
  plus the paired `.emul` weights. Schema 1 and schema 2 files are refused
  with a migration instruction. The function returns
  `(model, pgeom, geom, info)`. Here `pgeom` is the parameter-geometry object
  that encodes model inputs, and `geom` is the output-geometry object that
  decodes model outputs. The `info` mapping carries intrinsic-alignment,
  polynomial-chaos-expansion, transfer, and family-specific facts. "HDF5-only
  reconstruction" means that no code defaults or external training files are
  consulted; the paired weights file remains required.
- Head artifacts rebuild from saved files alone. The family
  geometries rederive their split by calling `attach_head_coords` inside
  `_rebuild_model`. CosmoLike `DataVectorGeometry` instead persists the split:
  `state()` writes `bin_sizes` and, when present, `pm_kept` after
  `build_shear_angle_map` attaches them. This follows the additive
  `section_sizes`/`probe` pattern. Constructor attributes remain unset when
  their arguments are `None`, preserving the `hasattr` guards. A head artifact
  without this persisted split is refused with a bin-split migration message.
  Rebuild never rederives the CosmoLike split because doing so would require
  data files below `ROOTDIR`, the environment variable naming the active
  CoCoA project root, during inference.
- Acceptance evidence: save -> rebuild -> bitwise-equal prediction,
  plus a drift test that monkeypatches a sharp code default
  (`make_activation` `n_gates` 3 -> 7) and requires the rebuilt prediction to
  remain unchanged. `ai/tests/test_model_recipe.py` removes each
  required recipe field in turn, distinguishes explicit null from absence,
  compares the registry with all six live constructor signatures, and proves
  that the production validator imports no executable model code. Adding a
  model constructor field without adding its saved
  representation must fail that census. Hard-coded duplicates such as
  `compile_mode`, and read-side keys such as `eval_bs` that no writer persists,
  are forbidden drift channels.
- Geometry classes live in
  `emulator/geometries/{parameter,output,scalar,cmb,grid,grid2d}.py`.
  Retired flat module paths remain absent. Loading a retired flat path raises
  `ModuleNotFoundError` naming that path. The `geo-paths` gate requires package
  markers in every new save, refusal of retired paths, and a repository-tree
  census. Every save writes folder paths through `type().__module__`.

## Saving one artifact pair

**Rule.** One saved emulator is the exact `<root>.emul` weights beside the
matching `<root>.h5` scientific record. `save_emulator` validates every
required input (recipes, facts, composition, geometry widths, head layout)
before it writes anything, writes both members to temporary names, and
renames them into place only when both are complete, so a crash mid-save
leaves no partial file under the public names. An occupied root — a complete
pair, either lone member, or a symbolic link — refuses before any temporary
file is created and stays byte-for-byte unchanged: a training run never
replaces an existing emulator.

**Reason.** The two files have different formats but one scientific meaning.
Writing either destination directly can leave new weights beside an older or
truncated record; silently replacing a saved emulator would change what an
existing chain's results meant.

**Read-side rule.** `rebuild_emulator` loads the checkpoint with
`weights_only=True` through a loader that accepts only tensor values, so a
rewritten checkpoint cannot execute pickle payloads or masquerade a
non-tensor mapping as a model state. Every saved string passes its closed
schema (the model recipe, composition record, and scientific record) before
any dynamic import or weight load. The user owns keeping a pair's two files
together; the library does not carry a cryptographic pair binding.

**Acceptance evidence.** A valid save rebuilds and leaves exactly the two
public members; complete, partial, and symbolic-link roots refuse before
staging with their bytes preserved; an injected HDF5 failure removes every
temporary file; an unsafe pickle value and a non-tensor checkpoint refuse
before model construction (`ai/tests/test_results_artifact_pair.py`).

### Run tags name output files readably

The run tag appended to the `--save` and `--diagnostic` roots is a readable
label — `<model>[_t<T>]_ntrain<N>` — not a scientific identity. The
scientific description of a saved run lives inside the artifact: the resolved
config, the model and training recipes, the staged-selection records, and the
`.facts.yaml` scientific record. Two runs the tag does not distinguish (for
example CMB `TT` versus `EE` under one model and row count) need different
`--save` names; the occupied-root refusal makes a collision loud instead of
silent.

## Inference: `EmulatorPredictor` and the five Cobaya adapters

`EmulatorPredictor` in `inference.py` owns prediction physics. Each class in
`cobaya_theory/` is a thin adapter: it contains no `nn.Module` and duplicates
no prediction physics. A Markov-chain Monte Carlo (MCMC) configuration written
in YAML names artifact path roots rather than
repeating architecture or whitening information. Geometry names determine
the required parameters, and `model_recipe` replaces the retired `extrapar`
convention; neither fact has a second manually maintained list.

The five adapters expose distinct products. `emul_cosmic_shear` returns a
data vector with `dv_return: section|3x2pt`; `3x2pt` means the full combined
three-probe weak-lensing layout, while the default `section` mode lets
the likelihood join per-probe sections, using `section_sizes` and `probe`
persisted in the geometry. `emul_scalars` returns derived parameters.
`emul_cmb` implements `get_Cl`. `emul_baosn` provides piecewise Hubble and
distance products. `emul_mps` provides the power-spectrum grid and
interpolator used by the `EMUL2` theory component in the shipped two-emulator
Cobaya configuration. Each adapter refuses an incompatible artifact and
names the adapter that owns that artifact kind.

Cobaya loads an external `Theory` class through `python_path`, not `path`.
Without `python_path`, an incompatible adapter bundled in the CoCoA Cobaya
fork can shadow the intended class. Configuration examples must therefore use
`python_path` for these external adapters.

Each family branch in the predictor reuses the training decoder. Factored
intrinsic alignment calls `TemplateFactoredChi2.decode`; neural polynomial
chaos uses its residual or ratio decoder; CMB uses the amplitude-law decoder;
grid output returns `{"z", quantity}`; and grid2d returns law-space
`{"z", "k", surface}`. The MPS adapter, rather than the predictor, restores
the Syren analytic base. Transfer dispatch precedes intrinsic-alignment
dispatch because a factored transfer correction would otherwise enter the
wrong branch.

Analytic-rescale runs are outside the predictor contract unless an artifact
persists everything needed to reconstruct the parameter-dependent inverse
transform. The rescale section below defines the refusal and extension rules.
Geometry whitening tensors inherit float64 values; because the Apple Metal
Performance Shaders accelerator backend does not support float64,
matter-power-spectrum inference may require an explicit downcast. CPU and
CUDA are the documented targets. A downcast converts a value to a data type
with fewer representable bits, such as converting float64 to float32. For
direct scripting without Cobaya, the README appendix “Scripting a saved
emulator” describes the two-door
`EmulatorPredictor` pattern. Background-family use pairs it with
`emulator.background.distance_interpolators`.

### Cobaya dependency and returned-value contracts

Artifact-only adapter tests are not a substitute for a real Cobaya
dependency-resolution test. Two dependency contracts therefore have live
Cobaya coverage:

- `emul_mps.get_can_support_params()` returns an empty list because that hook
  names sampled inputs a component can own; it does not advertise calculated
  products. The `get_Pk_grid` and `get_Pk_interpolator` methods advertise the
  two power-spectrum products, while `get_can_provide_params()` advertises
  `sigma8`, the root-mean-square matter-density fluctuation within spheres of
  radius 8 megaparsecs divided by `h`. A gate that replaces Cobaya's `Theory`
  class with a small test substitute and manually assigns `output_params = []`
  cannot establish this routing.
- When `want_derived` is true, `emul_scalars.calculate()` creates a new
  `state["derived"]` mapping when a direct caller did not supply one, retains
  any existing derived values, and publishes the artifact outputs there. The
  optional `provides` list checks names but never filters that artifact union.
  A real Cobaya construction asks a likelihood for one scalar and requires
  that value to travel through the same derived-result route.

The shared Syren reader accepts either `As_1e9`, which represents
`10^9 A_s`, or `As`, which represents the dimensionless primordial scalar
amplitude `A_s`, and canonicalizes a consistent pair. Requirement construction
must follow the artifact parameter names and that alternative-name rule. An
artifact that names `As_1e9` must
not acquire a redundant `As` requirement merely because it uses a Syren law.
The EMUL2 example may define a derived `As` bridge for another component, but
the bridge is not part of the adapter's scientific requirement.

<a id="adapter-contracts-strict-inputs-and-composition"></a>
#### Focused input and cosmic-shear composition evidence

`adapter-contracts.strict-inputs-and-composition` checks the extra_args
refusals below, then constructs concrete cosmic-shear section plans. It
requires disjoint sections to follow physical block order and requires
overlaps, incompatible layouts, repeated full vectors, and wrong widths to
stop before they can become a likelihood vector.

#### Adapter values, multi-emulator assembly, and CMB requests

**Rule.** Each adapter checks its own extra_args inline, in a few direct
lines. An unknown key is refused loudly, naming the accepted keys and the
retired legacy ones. `emulators` must be a nonempty list, with exactly two
entries for BAOSN or MPS. A relative root is joined onto `ROOTDIR`. The
device pick resolves `cpu` / `cuda` / `mps` and falls back toward the CPU
when the requested accelerator is unavailable. Values are otherwise read
with the plain YAML types the sampler configuration produces; the deleted
shared strict-value validator module is not restored (ruled
over-engineering and removed).

**Composition rule.** Multi-emulator mode must never concatenate full-vector
predictions blindly. Under `dv_return: 3x2pt`, multiple predictors are refused
unless one global vector is assembled after proving compatible layouts and
disjoint blocks. Section mode requires compatible stored layouts and unique,
non-overlapping probe blocks. Duplicate roots or probes, and a `3x2pt`
artifact combined with any constituent probe, fail before prediction. A valid
disjoint multi-probe case preserves its defined ordering. Blind
`np.concatenate` would turn two full vectors into one length-`2N` vector or
serve an overlapping likelihood block twice.

**CMB request rule.** `must_provide` requires the angular-spectrum (`Cl`)
request to be a mapping, requires every requested spectrum to be one a
loaded artifact provides, and requires the requested maximum multipole to
sit inside the artifact's stored range — an emulator has no accuracy beyond
its training grid, so an out-of-range request is refused rather than
truncated or zero-padded.

**Acceptance evidence.** The adapter-value checks must refuse an unknown
extra_args key, a missing or empty `emulators` list, and a one-root list
handed to a pair adapter. Composition checks must refuse two full vectors,
duplicate sections, and overlapping sections while accepting one disjoint
section pair. CMB request checks must refuse a non-mapping request, an
unknown spectrum, and an out-of-range `lmax`.

The MPS pair validator enforces the serving tuples
`pklin/Mpc3/(none|syren_linear)` and
`boost/dimensionless/(none|syren_halofit)`. The read side rejects a malformed
or hand-built HDF5 record instead of interpreting an unsupported
quantity-and-law combination as raw output.

**Live Cobaya evidence.** One small real Cobaya construction uses synthetic
artifacts rather than a stubbed base class. Dependency resolution must assign
power-spectrum (`Pk`) products to the MPS theory, register `sigma8` as a
derived value, call the scalar adapter with `want_derived=True`, and place the
advertised outputs in the returned state. Separate refusal cases cover an
invalid MPS law-and-units tuple and an `As_1e9`-only configuration with no
redundant `As` bridge.

### Geometry state and covariance guards

**Reason.** A class marker identifies the constructor but does not prove that
the saved tensors describe a finite, invertible transform. A zero scale,
nonpositive covariance eigenvalue, malformed basis, duplicate destination
index, or inconsistent dimension can preserve model-weight shapes while
producing nonfinite values or the wrong coordinate map. Clipping a negative
eigenvalue to zero merely changes an invalid covariance into a later division
by zero.

**Implementation boundary.** The shared geometry validator runs in
`ParamGeometry.from_covmat`, every sample- or log-parameter builder,
amplitude-factor and warm-start construction,
`DataVectorGeometry.from_cosmolike`, and every `from_state` rebuild. The
inverse covariance is stored as `Cinv`; `sqrt_ev` stores covariance
eigen-scales; and `dest_idx` maps kept entries into the full vector.

**Rule.** The validator checks shapes, finite values,
unique/in-range indices, monotonic finite axes, positive scales/eigenvalues,
orthonormal bases within a documented tolerance, covariance symmetry and
positive definiteness, plus family registry/units tuples. Apply it on both
training construction and HDF5 rebuild before tensors feed a model. Do not
silently floor or clip a scientific covariance: reject it with the smallest
eigenvalue and source/bin name. Gates cover a singular covariance block, a
tiny negative eigenvalue, a zero scale in a same-shaped h5, duplicate
`dest_idx`, and a valid ill-conditioned symmetric positive-definite matrix
just above the stated tolerance.

## Public inference validates numerical inputs and outputs

**Reason.** Names and lengths alone do not validate a numerical input. A
Boolean, not-a-number value (`NaN`), infinity (`Inf`), or nonscalar value can
enter parameter whitening and
propagate through a model. Decoding can likewise produce a nonfinite or
wrong-shaped result that looks structurally valid to an adapter. The CMB
factor `exp(2 * tau) / A_s` adds a family-specific domain: the scalar
amplitude `A_s` must be positive and the optical depth `tau` must be finite.

**Rule.** `EmulatorPredictor._as_row` requires each supplied value to be a
finite real scalar and refuses Booleans, naming the stored parameter. The
parameter geometry (`pgeom`) must return finite encoded values. Model output
and every decoder branch must have the exact expected shape and finite values
before conversion to NumPy, a dictionary, or a scattered vector. The CMB
amplitude law requires finite `tau`, strictly positive finite `A_s`, and a
strictly positive finite factor. BAOSN and MPS keep their own positivity
rules. No positivity rule applies to `TE` or a generic scalar output.

**Implementation boundary.** `_as_row` owns input validation;
`CmbFactoredChi2._factor` owns the CMB domain; and `EmulatorPredictor` owns
post-encoding, post-model, and post-decoding checks shared by all adapters.

**Acceptance evidence.** A finite control remains bitwise unchanged. Mapping
and ordered-array inputs containing a Boolean, NaN, or infinity must refuse
and name the parameter. Separate cases place a nonfinite value after encoding,
model evaluation, and decoding and require the stage to be named. CMB cases
cover nonpositive or nonfinite `A_s`, nonfinite `tau`, and an overflowed
factor. A wrong output width must refuse before reshape or scatter. Scalar,
CMB, grid, grid2d, and data-vector return branches are all exercised.

## Fine-tuning across every family

`train_args.finetune` accepts `from` and optional `compile_mode`. The source
HDF5 record owns the architecture; a sibling `model` block is refused. A lower
learning rate is selected through the ordinary `lr` block. One decade below
the source rate with `warmup_epochs >= 3` is the teaching recommendation
because the optimizer moments start cold.

**Epoch-zero rule.** Before the first training update, the warm-started model
computes the source function independently of the added parameters. The parity
gate requires `max|dv| <= 1e-5` in float32 and exact zero for equal-name runs.

**Implementation boundary.** `warmstart.py` extends the parameter geometry in
blocks. Shared rows keep the source rotation bit for bit. Added coordinates
use their marginal covariance block, so the encoded layout is
`[shared; extras; raw amplitudes]`. Equal-shaped tensors copy verbatim. Let
`n_x` denote the number of added input coordinates. When input dimension one
grows by exactly `n_x`, the source columns are followed by exact-zero added
columns. The source artifact pins the output-geometry class
and state. Cross-correlations between added and shared inputs are deliberately
not whitened because doing so would change the shared encoding.

Scalar, CMB, grid, and grid2d fine-tuning pin the source output geometry after
family-specific compatibility checks. A wrong family or mismatched metadata
is refused.

**Anchor rule.** Let `W` be a current trainable parameter, `W_0` the matching
saved source parameter, `lr` the current learning rate, `lambda` the configured
anchor strength, and `mask` the binary selection mask. An enabled fine-tune
anchor applies the decoupled post-step update
`W <- W - lr*lambda*mask*(W - W_0)`. Added input columns carry the new physics
and receive mask zero. The anchor is not part of the scalar loss,
because Adam, an adaptive optimizer that rescales updates using stored first
and second moments, would rescale it through those moments. A nonzero anchor
is normally paired with `weight_decay: 0.0`.

Saved provenance uses `finetuned_from` and `finetune_extra_names`.

### Warm-start source reads and perturbed finite values

`FinetuneSource` is one in-memory object. A successful construction opens
the source HDF5 file twice. `rebuild_emulator` owns the first open,
reconstructs both geometries and the network, and loads the `.emul` weights
once. `load_source` owns the second HDF5 open because the warm-start validator
needs the model recipe, saved rescale value, and resolved data block that
`rebuild_emulator` does not return. The class and loader docstrings teach
that sequence directly. The class attribute list includes `ia` and
defines its three values: `nla`, `tatt`, and `None`.

Both parity paths name and screen the two values produced only after the
extra-coordinate perturbation:

1. Fine-tuning screens `enc_pert` as `perturbed encoded new-run inputs`, then
   screens `out_pert` as `perturbed epoch-0 new-model outputs`.
2. Transfer screens `enc_pert` as `perturbed encoded run inputs`, then
   screens `composed_pert` as `perturbed epoch-0 composed prediction`.

All four calls use `_require_parity_finite`. Its shared error names the
pipeline side, the quantity, and the staged source-row coordinates. The
comparison runs only after both values are finite. The baseline path and
parity tolerances are unchanged.

**Acceptance evidence.** `ai/tests/test_warmstart_perturbed_finite.py` must
include two finite controls that retain the fine-tune and
transfer parity verdicts. A NaN appears only in row 9 of each perturbed
encoding. An Inf appears only in row 9 of each perturbed output. Every
failure names the required side, quantity, and row. Four mutation tests skip
one guard at a time: skipping an input guard changes the reported quantity to
the later output, while skipping an output guard restores the misleading
`extra parameters leaked` or `extra parameters moved` diagnosis. Each
mutation test first verifies that the targeted production guard executes.
Disabling or deleting that guard must then make the mutation test fail.

The corresponding gate coverage is:

- `ai/gates/checks/finite_contract.py` Part D receives the fine-tune
  perturbation-only NaN/Inf legs and both skip-one-guard mutation arms.
- Part E receives the matching transfer legs and mutation arms.
- The documentation-examples gate receives the runtime read census: a real
  tiny artifact must produce two `.h5` opens, one
  `.emul` load, one returned `FinetuneSource`, and a constructor-field census
  that includes `ia`.

These requirements are acceptance only when the registered gate executes the
legs; a local log or an unrelated passing child is not a substitute.

## Fine-tune anchor behavior

**Availability boundary.** `train_args.finetune.anchor` is refused unless the
complete anchor contract in this section is implemented and its registered
gate passes. `validate_finetune_config` therefore raises for the key rather
than advertising an update that may match no parameters.

**Reason.** Eager execution means an ordinary, uncompiled PyTorch module.
Source reference tensors and masks use eager-module parameter names. A
compiled live model can expose `_orig_mod.`-prefixed names. If
`build_anchor` merely skips absent reference names, a positive anchor can
produce zero entries and no update while the artifact still records the
configured value.

**Rule for enabling the feature.** The anchor strength is a finite,
nonnegative real; NaN and either infinity are refused. Compile prefixes are
canonicalized at one boundary, or anchoring operates on the underlying eager
module, with exact one-to-one parameter coverage. A positive strength with
zero matched trainable parameters is refused. The implementation reports and
validates matched, masked, frozen, and unexpected names. Masks use the same
canonical mapping. The artifact records executed anchor evidence, including
matched count and effective strength, rather than configuration alone.

**Acceptance evidence.** A real `from_config` run proves that a positive
anchor reaches training. One eager and one compiled step match the analytic
update. Added columns remain free under their masks. Missing or extra reference
names refuse. Strength zero is an identity. Transfer-refine exercises eager
coverage. A positive strength records a nonzero matched count, and artifact
readback verifies the resolved record.

### Fine-tune anchor teaching contract

**Documentation rule.** The root README must explain the fine-tune anchor as
weight-displacement regularization, distinguish it from ordinary weight
decay, identify the selected parameters, and show the executed update. A
description such as “a pull back toward the saved weights” is insufficient on
its own.

Let `W_{j,0}` be parameter tensor `j` in the saved source emulator, `W_j` its
current fine-tuned value, and `M_j` an element-by-element binary mask. L2
starting-point regularization (`L2-SP`) penalizes squared distance from the
saved starting parameters. Its mathematical form is

```
R(W) = (lambda / 2) sum_j || M_j * (W_j - W_{j,0}) ||_F^2 .
```

Define every symbol in prose: `lambda` is the YAML `anchor` strength; `*` in
this equation means element-by-element multiplication; and the squared
Frobenius norm means “square every selected displacement and add the squares.”
Explain that ordinary weight decay pulls `W_j` toward zero, whereas this
quantity measures movement away from an already-trained source emulator.

Then state the executable truth. CoCoA SONIC deliberately does **not** add
`R(W)` to the scalar scientific loss seen by AdamW, the adaptive optimizer
that performs the ordinary parameter update and applies decoupled weight
decay. After the ordinary
optimizer step proposes `W_j^opt`, `Anchor.apply` performs the decoupled,
in-place update

```
W_j^+ = W_j^opt
        - eta_j lambda M_j * (W_j^opt - W_{j,0}),
```

where `eta_j` is the current learning rate of that parameter's optimizer
group. This keeps Adam's adaptive second-moment rescaling out of the anchor.
There is no hidden division by batch size, layer width, or number of
parameters. `anchor: 0.0` is an exact no-op. For a one-number example, if
`W_0 = 2.0`, the optimizer proposes `W^opt = 2.4`, `eta = 0.01`, `lambda =
0.5`, and the mask is one, the anchor removes `0.01 * 0.5 * 0.4 = 0.002` and
stores `W^+ = 2.398`. This example must say that the optimizer's scientific
step happened first; the anchor modifies the resulting displacement.

The parameter-selection paragraph must say that matched, trainable source
parameters are anchored: matrices, biases, affine-normalization parameters,
and trainable activation parameters all qualify when they are model
parameters owned by an optimizer group. Geometry tensors and other
non-parameter state do not. When fine-tuning adds cosmological inputs, the
newly appended input-weight columns start at zero to preserve epoch-0 parity
and receive mask value zero, so they remain free to learn the new physics;
the pre-existing columns and other matched trainable parameters receive mask
value one. Recommend optimizer `weight_decay: 0.0` beside a nonzero anchor,
because simultaneous weight decay pulls toward zero rather than toward
`W_{j,0}`.

The availability boundary stays adjacent and unambiguous. Ordinary
fine-tuning is unanchored while `validate_finetune_config` refuses
`train_args.finetune.anchor`. Documentation must not print a usable
fine-tune-anchor YAML unless the registered gate passes. The separate
cosmic-shear `transfer.refine.anchor` path anchors only the frozen base during
joint refinement, not the correction network. Diagonal-family transfer
refinement is refused.

**Acceptance evidence.** Root `README.md` contains the definition, executed update,
symbol definitions, one-number example, mask ownership, weight-decay
distinction, and availability boundary in the fine-tuning section; `emulator/README.md`
contains a shorter equation-and-owner pointer rather than duplicating the full
tutorial. The prose must never say the implementation adds an L2
term “to the loss.” A claim-consistency scan covers “anchor,” “L2-SP,” “penalty,”
and “refine” across both READMEs. This documentation check does not change
Python abstract syntax trees or runtime behavior.

## Transfer learning (family-wide, scalar excepted)

**Scope.** Transfer applies to CosmoLike, CMB, grid, and grid2d families.
Scalar transfer remains unsupported as an explicit product boundary rather
than a structural limitation.

**Reason.** A new physics sector may require capacity beyond an ordinary
same-size fine-tune. Transfer is judged by sample efficiency, meaning accuracy
per expensive training cosmology, rather than wall-clock time alone.

**Composition rule.** The trained base is frozen as a whole. A small parallel
correction network receives the complete new parameter space and produces a
correction `r`. Form `gain` computes `base * (1 + r)`; form `sum` computes
`base + r`. Composition occurs in either physical or whitened space. An
omitted space resolves to the form's documented recommendation and is saved
as an explicit value. The `model` block describes the correction network.

Diagonal families use `emulator/losses/transfer.py::TransferDiagChi2`, a
subclass of `CmbDiagonalChi2`. They accept plain bases and whitened space only
because physical composition is separated from their metric by an
elementwise scale and can cross a logarithmic-law domain. Both forms remain
available, with a zero-crossing notice for `gain` and `sum` recommended.
`transfer.refine` and roughness with transfer are refused. CMB requires
`amplitude_law: none` on both sides so only one target-construction rule owns
the data. Compatibility checks pin spectrum, multipole coordinates, scales,
redshift coordinates, quantity, units, and law as applicable. A cross-family
base is refused by `from_config`.

**Efficiency rule.** The frozen base runs once per row during encoding. The
staged target stores `[base; truth]`, and the repeated chi-square calculation
composes those cached values without evaluating the base again. Hook counts
in the acceptance check establish the one-evaluation property.

**Identity rule.** An exactly zero correction reproduces the frozen-base
decode bitwise when both sides use the same arithmetic path. The factored
physical path reassociates template combination and unwhitening, so it uses a
documented tolerance from `1e-6` to `1e-5`; the measured reference difference
is approximately `4e-6`.

**Artifact rule.** The `transfer_base` HDF5 group embeds the base recipe,
state, both geometries, form, and space. It is not an external reference, and
transfer chaining is refused. Optional stage-two `transfer.refine` unfreezes
the base once, applies per-group `base_lr_scale`, and requires an explicit
anchor strength, including `0.0`. A refined artifact keeps pretrained
reference weights `W_0` in `transfer_base` and prediction weights in
`drifted_state`. The states must permit exact drift recomputation.

The training driver gives the writer the live frozen base, or both the cloned
pretrained base and the live drifted base after refinement. Rebuilding selects
`state` or `drifted_state` from the explicit refinement fact and loads that
mapping strictly into the registered base model. Missing, unexpected, or
wrong-shaped tensors therefore refuse. The file does not hash the embedded
mapping again or copy that hash into configuration records. A same-shaped
value edit inside the HDF5 file is user responsibility.

The resolved transfer record names the source artifact's path root, the
transfer form, and the materialized space, so the saved run states what it
composed.

The four supported training modes are from-scratch training, anchored warm
start, frozen-base transfer, and anchored joint refinement. The decoupled
L2-SP anchor strength spans the refinement range from frozen to free.

### Cross-family refusal fixtures test one invalid condition at a time

A test fixture is a controlled input constructed for one acceptance check. A
refusal fixture must violate only the rule under test. The cross-family
transfer leg therefore saves a plain `GridGeometry` base and points a grid2d
configuration at that base. A transfer artifact would violate both the
no-chaining rule and the cross-family rule, allowing the no-chaining error to
fire first and hiding whether the family check works.

**Code ownership.** `ai/gates/checks/transfer_identity.py` owns the fixture.
The production family check remains in `_load_diag_transfer`; production code
must not be changed to accommodate an ambiguous fixture. The dedicated
no-chaining leg continues to test transfer-over-transfer refusal.

**Acceptance evidence.** The cross-family leg must raise a `ValueError` that
names the never-cross-families rule. The plain base must contain no
`transfer_base` group. The no-chaining leg must pass independently.
The CMB identity metadata must describe all five covariance checks in plain
language: exact contraction, pretrained-weight miss, raw-versus-scaled fixture
integrity, width-three band projection, and exact zero-band weight. A
GPU-capable environment must rerun both identity gates after a fixture change.

## Code ownership map

- Save/rebuild and schema ownership: `emulator/results.py` and
  `emulator/fixed_facts.py`.
- Model-recipe ownership: `emulator/model_recipe.py`.
- Ordered training-pass construction: `emulator/training.py::run_emulator`.
- Save-time history shape checks: `emulator/results.py::_history_arrays_for_save`.
- Public prediction ownership: `emulator/inference.py`.
- Fine-tune and transfer-source ownership: `emulator/warmstart.py`.
- Transfer composition ownership: `emulator/losses/transfer.py`.
- Adapter ownership: `cobaya_theory/`.
- Registered gates: `save-rebuild-drift`, `cobaya-adapter`,
  `finetune-identity`, `finetune-smoke`, `transfer-identity`,
  `transfer-smoke`, and `geo-paths`.

## What a saved recipe does not prove

**Reason.** A model recipe records a class name and its constructor values.
Those facts are enough to rebuild the supported model, but they cannot prove
that the Python implementation behind a stable name has never changed. A
manually maintained label such as `model:...:v1` would only repeat another
name unless every scientific code change updated it correctly.

**Rule.** Save the concrete facts the prediction uses: model recipe, geometry
state, analytic-law name, composition mode, scientific fixed facts, and Git
commit provenance. Do not add a second compatibility registry that copies
those facts and claims to authenticate implementation behavior. If a model or
analytic formula changes incompatibly, update its direct saved format or
regenerate the artifact rather than relying on a manually bumped label.

`git_commit` remains provenance rather than an execution lock. A user who
loads an artifact with edited local model or formula code is responsible for
checking that provenance and regenerating scientific results when needed.

**Acceptance evidence.** New root and transfer artifacts contain no duplicate
compatibility-manifest dataset. Complete model recipes still reject missing or
unknown constructor fields, geometry and composition facts remain explicit,
and ordinary and transfer save-to-rebuild predictions remain unchanged.

## Artifact composition is declared, not inferred

**Reason.** Optional HDF5 group presence alone cannot establish scientific
composition. Deleting an NPCE `pce` group can leave same-shaped weights that
strict-load under an ordinary decode and return finite but different values.
A discriminating numerical example changes the Hubble constant from
`69.68846130371094` to `66.8885269165039` and the matter-density fraction from
`0.31317755579948425` to `0.30848246812820435`; the missing PCE contribution
then changes the finite prediction. Weight-pair identity and recipe-key
totality therefore do not replace an explicit composition fact.

**Rule.** Persist a native required composition enumeration, meaning exactly
one value from the closed set `plain`, `npce`, or `transfer`, from the executed
run. Store transfer-refined state as a separate native fact. Validate mode
against the exact required and forbidden group
set in both directions before constructing a model. Authenticate the HDF5
scientific payload/manifest so deletion cannot leave a valid marker. The
resolved consumed record retains NPCE/transfer facts. Schema-v2 absence never
means plain; legacy presence-only artifacts refuse with a migration/re-save
instruction. Apply the same contract to `pce` and `transfer_base`, including
mutual exclusion.

`config_yaml` may retain the top-level `pce` or `transfer` block, but provenance
YAML cannot substitute for the authoritative native composition facts. The
acceptance check deletes the parent group while preserving the refined marker
and configuration, then requires refusal. One authoritative consumed mode
owns runtime validation; provenance
YAML is corroborating evidence, not a second inference algorithm.

**HDF5 presence inventory.** The production census contains the required
`model_recipe`, optional `pce`,
optional `transfer_base`, nested `drifted_state`, and warm-start's
`transfer_base` no-chaining branch at the HDF5 membership layer. The
composition contract owns the composition and refined-state cases.
Conditional geometry-state keys are
`section_sizes`/`probe`/`bin_sizes`/`pm_kept`, CMB `as_ref`/`tau_ref`, Grid2D
`const_mask`, and AmplitudeFactor `names`. Every item except `const_mask`
either has an explicit governing fact and refusal, raises before its requested
consumer path, or is already owned by recipe/geometry totality. Only deletion
of `const_mask` silently reinterprets a valid artifact; the corresponding
contract lives in `families-background-mps.md`. Cobaya adapters perform no
separate direct HDF5 presence dispatch.

**Acceptance evidence.** Save, forge, and rebuild checks cover valid plain,
NPCE, frozen-transfer, and refined-transfer controls. Separate negative checks
delete `pce`, delete `transfer_base`, change the enumeration without changing
the groups, add a forbidden second composition group, and delete or forge the
refined half. A mutation that restores presence-only inference must
strict-load a network with the same shape and still fail the known-answer
prediction. The PyTorch-and-HDF5 checks run in the project GPU environment.
Enumeration-to-group validation also has a CPU-only HDF5 check.

<a id="artifact-composition-contract"></a>
**Artifact composition is authoritative before construction.** The writer
derives and persists native root facts
`composition_mode` (`plain`, `npce`, or `transfer`) and `transfer_refined`
from the executed run. Callers cannot replace either fact through `attrs`.

The resolved record carries top-level `composition_mode`, `transfer_refined`,
`pce`, and `transfer`. The raw config corroborates any non-null `pce` or
`transfer` block, but neither YAML surface can substitute for a missing root
fact. A presence-only artifact receives an explicit re-save/migration refusal.

Immediately after the scientific schema record is read, rebuild validates the
four legal rows: plain has neither base group; NPCE has only `pce`; frozen
transfer has only `transfer_base`; refined transfer additionally has
`transfer_base/drifted_state`. Every required and forbidden edge is checked in
both directions before recipe parsing, geometry/model construction, or
`torch.load`. Rebuild, inference, and warm-start then route on the validated
enumeration, never on optional-group presence.

The CPU/HDF5 gate proves the four valid rows, thirty focused forgeries,
writer/read agreement on the native form-and-space grammar, and the
pre-construction call order. The child command emits one final evidence result
named `artifact-composition.contract` for that assertion. Whole-pair
cryptographic binding remains the
separate artifact-integrity unit; this enumeration contract does not claim that an
attacker cannot rewrite every corroborating HDF5 surface together.

## `config_resolved_yaml` records what the run consumed

**Reason.** Raw inputs do not describe the values a run consumed. An absent
loss block consumes the default square-root mode, and omitted reverse-Huber
(BerHu) loss knots consume materialized numeric values. Each phase resolves
effective loss, BerHu knots, exponential-moving-average (EMA) settings,
trimming, focus, clipping, rewind, learning rate, warmup, and scheduler values.
Transfer refinement inherits another effective pass. A record containing only
raw trunk/head overlays, one top-level scheduler, or
`{epochs, base_lr_scale, anchor}` would require a reader to rerun default and
inheritance logic. History rows can also include refinement while a
pre-refinement epoch count does not.

**Rule.** Persist a `passes` sequence in execution order. Every entry records
the phase name, model training phase, epoch count, computed learning rate,
warmup, scheduler class and resolved keyword arguments, fully resolved loss,
trimming, focus, clipping, rewind, and EMA settings. Transfer refinement has a
separate pass entry with every inherited effective value. Persist run-level
roughness separately when one value configures the loss object for the whole
run. Persist `total_epochs` and the five history arrays. Keep the raw YAML
separately as provenance. The artifact writer checks only that the history
arrays are finite and have compatible shapes. Prediction does not reconstruct
training: reopening treats the pass plan and history arrays as provenance.

**Implementation boundary.** The shared training configuration resolver owns
the complete pass records in `emulator/training.py::run_emulator`. The artifact
writer persists those records and `total_epochs` without becoming a second
training-policy engine. `emulator/results.py::_history_arrays_for_save` performs
the narrow publication check on array finiteness and shape. Reconstruction
does not read the `history` group or validate pass grammar.

A two-phase run with two trunk epochs and three head epochs records contiguous
history slices `[0, 2)` and `[2, 5)`, followed by `total_epochs: 5`. Four saved
loss rows cannot describe that run and must refuse. A transfer-refinement pass
appears after the ordinary or trunk/head passes and extends the same contiguous
history instead of starting an unrelated counter.

**Acceptance evidence.** `ai/tests/test_training_pass_recipe.py` proves
that an absent loss block records the consumed square-root mode rather than
null; omitted BerHu knots appear as consumed numeric values; trunk and head
overrides produce two complete pass records; a null EMA setting records that
EMA is disabled for that phase; refinement becomes a third complete record;
and each pass names its exact history slice.
`ai/tests/test_artifact_recipe_preflight.py` proves that incompatible or
nonfinite histories fail before publication. It also removes the entire
history group from a valid saved artifact and requires model reconstruction to
succeed, proving that historical curves are not prediction inputs.

### Transfer-refine drift measures trainable parameters only

**Reason.** A relative weight-drift metric must not include persistent,
non-trainable buffers such as `pad_idx` layout indices from the residual
convolutional (`ResCNN`) and residual transformer (`ResTRF`) designs, or fixed
PCE buffers. Such buffers contribute zero to the numerator but can inflate
the denominator and dilute the reported change. Relative drift is also
undefined when the reference norm is zero; a moved zero tensor must not be
reported as relative drift `0.0`.

**Rule.** Define the metric over trainable parameters only through an explicit
canonical key set, excluding buffers, layout, and other state. Persist the
numerator and reference norms, or absolute drift plus a named status, beside
any relative value. If `||W0|| == 0`, report exact zero only when the drift
norm is also zero. Otherwise report the absolute drift and a
`zero-reference` status, never relative `0.0`. Verify parameter-key equality
between the two states before publication. The declared key set and the two
persisted states must reproduce the stored summary exactly. If the metric
includes non-parameter state, name it state drift rather than weight drift.

**Acceptance evidence.** A large fixed integer buffer plus one moved parameter
matches a hand calculation and is invariant to buffer magnitude. An unchanged
zero-reference parameter reports exact zero; a moved zero-reference parameter
reports absolute drift and status. Missing or extra keys refuse before
publication. A multi-parameter known answer and artifact readback both
recompute the summary.

## Syren parameter aliases must agree

**Reason.** The Syren starting formulas read two equivalent amplitude names,
`As_1e9` and `As`, and several equivalent dark-energy coordinate forms. Here
`w` and `w0` name the present-day dark-energy equation-of-state value, `wa`
describes its time evolution, and `w0pwa` means `w0 + wa`. Silently choosing
one repeated value can make the network and analytic starting formula describe
different cosmologies.

The generator receives Cobaya's complete calculated input mapping. The
matter-power adapter asks Cobaya for the saved present-day name and calculated
`wa`, then rebuilds all four dark-energy names before prediction. Acceptance
cases must use those real routed inputs. The shipped EMUL2 evaluation
configuration supplies both amplitude names, so dual-amplitude input is also
a public configuration shape.

The numerical failure remains finite and can evade downstream checks. For
`{As_1e9: 2.1, As: 9e-9}`, silently choosing `As_1e9` changes the analytic
linear-power baseline by a maximum relative difference of `0.7667` compared
with using `As`. For `{w: -1.0, w0: -0.7}`, silently choosing `w` changes the
baseline by a grid-dependent maximum relative difference near `0.2449`. A
law-correction artifact is especially vulnerable because the network reads
the artifact's stored parameter names while the analytic base could follow a
separate alias-precedence rule.

**Rule.** Keep one Syren input path:

1. `resolve_dark_energy_coordinates` in `emulator/syren_base.py` is the shared
   authority for `w`, `w0`, `wa`, and `w0pwa`. `syren_params_from` remains the
   one entry point for the seven Syren arguments and delegates the last two
   values to that resolver. The complete law and completion rules live in
   `ai/notes/families-background-mps.md`.
2. `As_1e9` alone and `As` alone are accepted. When both are present, require
   `As_1e9 == 1e9 * As` within a documented float-representation tolerance.
   Otherwise raise an error that names both values and the conversion.
3. `w` and `w0` are two names for one present-day value. Either name may be
   used when the same input also supplies `wa` or `w0pwa`, or when the explicit
   saved law supplies the missing coordinate. When both aliases are present,
   require them to agree and name both values in a mismatch error.
4. `w0pwa` means `w0 + wa`. A complete transformed input supplies a
   present-day alias and `w0pwa`; the resolver derives `wa`. If `wa` is also
   present, require the supplied and derived values to agree.
5. Incomplete coordinates never select a law. `w0wa-cpl` supplies no missing
   value. `constant-w` supplies `wa = 0` only after a present-day value is
   supplied. `cosmological-constant` supplies `w0 = -1, wa = 0` and checks
   every repeated value against that pair.
6. Canonicalize only after every repeated value agrees. Never prefer one
   conflicting alias or silently replace missing evolution information with
   zero.
7. Do not duplicate the individual-value contract. The generator validates
   values when a sample enters generation, and the adapter validates values
   received from Cobaya. This rule owns relationships among aliases and
   transformed coordinates.
8. Requirement construction does not request a redundant amplitude alias. For
   time-varying dark energy, the adapter requests the present-day alias and
   calculated `wa`, not dropped `w0pwa`, then rebuilds every saved spelling
   before prediction.
9. On failure, `emul_mps.calculate` leaves no `Pk_grid`, interpolator, or
   derived state key. The generator refuses the sample before writing a raw or
   starting-surface row.
10. Documentation defines `As_1e9 = 10^9 As`, defines `w` and `w0` as two
    names for one present-day value, and defines `w0pwa = w0 + wa`. It never
    describes correctness as one alias being "preferred."

**Dark-energy acceptance evidence.** Direct `(w or w0, wa)` and transformed
`(w or w0, w0pwa)` inputs produce the same nonzero-`wa` pair. Conflicting
aliases or sums, a time-varying input missing one coordinate, and an incomplete
input with no explicit saved law refuse before generator or adapter output.
Explicit constant-`w` and cosmological-constant controls supply only their
documented values. The real-Cobaya check sends `w = -0.9, wa = 0.2` to the
adapter from sampled `w = -0.9, w0pwa = -0.7`, and generation and serving give
Syren the same pair. An adapter refusal leaves no partial power result, and a
generator refusal leaves no raw or starting-surface row.

**Amplitude-alias work still required.** The amplitude rule above remains
mandatory, but the current `syren_params_from` does not yet compare both names
when `As_1e9` and `As` are supplied together. That behavior must not be
reported as accepted until focused generator and adapter tests prove that
consistent names pass and inconsistent names refuse before either Syren
formula runs.

## Geometry scales remain valid after conversion to the storage dtype

**Reason.** A relative scale check performed in 64-bit floating point does not
prove that the stored 32-bit scale remains positive. Absolute underflow can
turn a valid-looking pre-cast scale into exact zero, after which encoding
divides by zero. `ScalarGeometry.from_targets`, `GridGeometry.from_targets`,
and `Grid2DGeometry.from_stats` all cross this representation boundary.

Let `f = nextafter(float32(0), float32(1))`, the smallest positive 32-bit
subnormal number, approximately `1.4013e-45`. A subnormal number is smaller
than the smallest ordinary positive value of the data type and therefore uses
reduced precision near zero. For targets `[0, 0, f]`, the
64-bit center is approximately `4.6710e-46` and the population scale is
approximately `6.6058e-46`, so a purely relative check accepts it. Both values
round to zero in 32-bit storage. Targets `[0, f, f]` preserve a nonzero stored
center but still round the scale to zero. These examples show why the stored
representation, rather than only the pre-cast ratio, owns validity.

**Rule.** The `from_state` read-side rule and covariance square-root rules
remain in force:

1. Validate the values that will actually be stored, not only pre-cast
   estimates. Cast every produced center-and-scale pair through the declared
   artifact and model data type (`float32`) first.
2. Require a finite stored center and a finite, strictly positive
   stored scale for every pair; refusal names the column or grid
   coordinate through the owning error message.
3. Apply the relative-resolution rule in that same storage representation:
   require `scale > 8 * float32_epsilon * abs(center)`. The absolute
   underflow test and this relative-collapse test must both pass.
4. Grid2d classifies a post-cast-zero scale as constant before the
   partial-pin versus whole-surface-refusal decision. The pin and
   dead-dump policies operate on stored-representation facts.
5. The same representability rule applies to covariance
   square-root scales: a positive
   float64 eigenvalue is not sufficient if its stored float32 square
   root is zero; refusal names the smallest stored scale.

**Acceptance evidence.** Scalar targets `[0, 0, f]` and `[0, f, f]` must
refuse before construction. A grid column with the same payload must refuse
and name its coordinate. A grid2d column is pinned consistently, while a
surface made entirely of such columns is refused as unusable. A scale just
above the stated storage boundary survives and round-trips bitwise through
`state` and `from_state`. A covariance eigen-scale that underflows during
storage is refused and names the smallest stored scale. NumPy arithmetic
checks run on CPU. Torch constructor and encoding cases join the registered
`scalar-identity`, `bsn-identity`, and `mps-identity` geometry checks in the
required GPU-capable environment.

## Scalar fine-tuning saves source provenance

**Reason.** Scalar fine-tuning loads a source, pins output geometry, and records
`_scalar`, `_finetune`, and `_finetune_root`. Its saved artifact therefore
needs the same source-provenance record as every other fine-tuned family. A
scalar save path that records only model, data, and best-metric attributes
cannot distinguish a cold run from a run whose architecture and initial
weights came from another artifact.

**Rule.** The fine-tune anchor rules remain in force:

1. One shared artifact-provenance assembler owns the common attributes
   for every training driver; scalar adds its family facts (outputs,
   training and validation parameter filenames) but never forks the
   fine-tuning, anchor, or source-provenance logic.
2. A cold run stores no fine-tune attributes. A fine-tune run stores the
   canonical resolved source identity and ordered extra parameter names.
3. The recorded source identity is the resolved root/digest actually
   loaded, never the raw YAML spelling.
4. An enabled executed-anchor record shares this same path and is never added
   to only one family driver.

**Acceptance evidence.** A scalar cold save omits fine-tune attributes. A
scalar fine-tune with equal parameter names saves `finetuned_from` and an
explicit empty extra-name list. Extended-input fine-tuning preserves the
ordered extra names. Scalar and one shared-driver family produce the same
common provenance schema. A mutation that removes the shared provenance call
must make artifact readback fail. The registered `finetune-identity` check in
`ai/gates/checks/` builds and reads the synthetic artifacts with Torch in the
required GPU-capable environment.

## Parameter covariances use the shared integrity contract

**Reason.** A nonempty one-parameter model is valid. NumPy loads a scalar
covariance file such as `# x` followed by `4.0` as shape `()`, not `(1, 1)`,
and `np.linalg.eigh` refuses that dimensional accident. The sample-derived
one-feature form from `np.cov(..., rowvar=False)` has the same zero-dimensional
shape. A multi-parameter covariance with a negative variance can instead reach
`np.sqrt` and create a NaN whitening scale. The same integrity boundary is
needed by `ParamGeometry.from_covmat`, sample-derived parameter geometry, and
`AmplitudeFactorGeometry.from_covmat`. This is separate from a one-row
parameter-table shape error: the parameter covariance itself is scientifically
valid when its one-dimensional representation is normalized correctly.

**Rule.** One shared covariance validator owns all parameter-side covariance
inputs:

1. Normalize covariance input to an exact two-dimensional square matrix before
   eigendecomposition. A valid scalar covariance becomes `(1, 1)`.
   Normalization never accepts malformed input; the normalized matrix still
   passes every integrity check below.
2. Header-name count, covariance width, and center width must
   agree exactly; a mismatch is refused naming all three observed
   dimensions.
3. Require finite, symmetric, strictly positive-definite values under the
   shared geometry-integrity policy at every parameter-side site. This check
   prevents `parameter.py` from taking the square root of a negative
   eigenvalue and producing NaN.
4. Apply the dimensional-totality rule wherever covariance is derived from
   samples with `np.cov`, including the one-feature case, and in
   `AmplitudeFactorGeometry.from_covmat`.
5. Preserve byte-for-byte numerical results for valid multi-parameter inputs.
6. Reuse the same validator at the `output.py` covariance sites so parameter
   and output geometry enforce one integrity policy.

**Acceptance evidence.** The arithmetic checks use CPU NumPy. Round-trip
checks use Torch in the registered `scalar-identity` gate:

- one header name plus a positive one-by-one covariance builds and satisfies
  `decode(encode(x)) == x`;
- the whitened displacement equals `(x - center) / sqrt(var)`;
- zero, negative, NaN, and Inf scalar variances are refused before the
  eigendecomposition or square root;
- header count, center width, and covariance dimension mismatches
  are refused with the three observed dimensions named;
- a normal multi-parameter control remains byte-identical;
- the sample-derived one-feature constructor, which is part of the public
  programming interface, receives the same one-dimensional acceptance test.

**User-visible result.** One-parameter emulators become buildable. Malformed
covariances are refused loudly instead of reaching training with a NaN
whitening scale.

## Artifact documentation teaches write/read symmetry

`results.py` gives a long inventory of HDF5 groups, but a first-time PyTorch
reader reaches the nested `write_state`, `_read_group`, `_rebuild_geometry`,
and `_rebuild_model` functions without the mechanics needed to audit them.
The distinction between a model object and its `state_dict`, an HDF5 dataset
and attribute, a tensor and a NumPy array, CPU and accelerator storage, and a
serialized class name and a live class object is assumed. Local comments alone
do not reveal the write/read inverse as one system.

**Rule.** The artifact-pair integrity documentation requires:

- Start with a two-column write/read table. For every payload, name the
  writer, stored representation, reader, rebuilt Python type, dtype/device,
  and owning constructor argument.
- Define `state_dict` as a name-to-tensor mapping of parameters and registered
  buffers, not a saved model object. Explain `detach` (remove gradient
  history), `cpu` (copy or move storage to host), `numpy` (share CPU tensor
  storage when possible), and why the saved tensor must not require the
  training GPU to load.
- Define an HDF5 group as a directory-like container, a dataset as an array
  payload, and an attribute as scalar metadata attached to a group or file.
  Show one concrete nested geometry state and its exact read-back dictionary.
- Explain recursion before `write_state` and `_read_group`: a nested dict
  creates a subgroup; the same function calls itself on that subgroup; the
  base cases are tensor, string list, numeric scalar, and serialized dtype.
- Explain dynamic class reconstruction step by step: split the persisted
  `module.Class` string, import the module, retrieve the class object, and call
  its `from_state` classmethod. State why `cls(...)` preserves subclasses.
- Explain strict weight loading: every expected key must exist and no
  unexpected key may remain. Explain why a compiled model's `_orig_mod.`
  prefix is removed before storage and compilation happens only after the
  eager object and weights are rebuilt.
- Draw the ordinary, NPCE, transfer, and refined-transfer ownership trees.
  Distinguish pretrained base weights used as reference from drifted base
  weights used for prediction.
- Prose must match executable behavior. Documentation may describe the
  `.emul` and `.h5` pair as one atomic publication only when the public writer
  and reader implement and test that property. Otherwise the direct-write
  behavior and its interruption limit are stated explicitly.
- Gates exercise the public save/rebuild pair and inspect the actual on-disk
  type, dtype, and device transitions. A test double is a small substitute
  object used only by a check. A test double for HDF5 recursion may teach
  structure but cannot substitute for a real-file round trip.

Completion includes concise docstrings for the nested helpers. The
documentation gives the reader one reversible map with enough mechanics to
verify every branch without copying the implementation into prose.

## Structured acceptance evidence

Each gate links every named assertion to one stable
HyperText Markup Language (HTML) anchor, a named location that a link can
target in this permanent note. The permanent-note evidence validator
refuses a missing anchor or a repeated evidence identifier. The artifact-side
gate anchors are listed here:

Every gate-summary block uses the same fields:

- **files** names the temporary or configured inputs and outputs;
- **subprocess** names the command that performs the checks;
- **metric** states the numerical comparison or refusal that decides success;
- **legs** lists the individually named assertions;
- **evidence** explains how command output proves each named assertion; and
- **capability boundary** names required software, scientific data, or
  hardware. A missing capability produces no passing evidence.

### fixed-facts-schema: a saved emulator records the science it was born under

`emulator/fixed_facts.py` stores two sibling blocks in a real HDF5 file.
`fixed_facts` records coordinates held constant while sampled coordinates
varied and is compared by equality. `input_domain` records the sampled support
and is compared by overlap. Keeping the blocks separate prevents per-key
exceptions to one ambiguous comparison rule. These checks need no accelerator.

<a id="fixed-facts-schema-record-round-trip"></a>
`fixed-facts-schema.record-round-trip` writes both blocks into a real file and
reads them back: the sampled names survive in order, a boolean fact comes back a
boolean (HDF5 has no Python types, and `True == 1` in Python), the fixed
cosmology survives, a sampled coordinate is absent from the fixed cosmology, and
two bounds differing in the last float32 digit stay distinct under the shortest
decimal that round-trips.

<a id="fixed-facts-schema-rewritten-record-refused"></a>
`fixed-facts-schema.rewritten-record-refused` proves the two-way check behind
"copied verbatim, never re-derived." The file carries the producer's own text
and the blocks parsed from it; the reader checks them
against each other in both directions. A fact edited in the stored block, and a
producer text swapped under blocks that no longer match it, are both refused,
and the refusal prints both sides.

<a id="fixed-facts-schema-missing-record-refused"></a>
`fixed-facts-schema.missing-record-refused` deletes each half of the record in
turn (either block, or the producer text) and requires the read to refuse with
the migration instruction named. A file that cannot say which cosmology it
belongs to is refused, not served.

<a id="fixed-facts-schema-legacy-version-refused"></a>
`fixed-facts-schema.legacy-version-refused` requires both legacy schema versions
(1 and 2) to refuse with the migration instruction, requires a block grammar
from the future to refuse, and requires the supported version to be accepted (a
check that only ever refuses proves nothing about the file it must let through).
The reader accepts exactly the schema version declared by
`emulator/fixed_facts.py` and refuses every other version with migration
guidance.

<a id="fixed-facts-schema-sampled-and-fixed-refused"></a>
`fixed-facts-schema.sampled-and-fixed-refused` requires a coordinate that is
both sampled and held fixed to be refused when the sidecar is composed,
naming the coordinate and both of its values. If it were allowed, the two
halves of the record would answer "what was w?" differently depending on
which half was read.

<a id="fixed-facts-schema-parameter-order-enforced"></a>
`fixed-facts-schema.parameter-order-enforced` requires a whitening geometry
whose parameter order is a permutation of the record's to be refused, with both
orders printed. A check that counted the names, or compared them as a set, would
let a permutation through, and a permutation silently pairs every incoming value
with the wrong parameter's column: the predictions are then confidently wrong
and nothing about the numbers looks unusual.

<a id="fixed-facts-schema-mutation-arms-red"></a>
`fixed-facts-schema.mutation-arms-red` breaks the record's own laws on
purpose and requires the guarding legs to go red: accepting a legacy schema
must fail the version leg, and a stored block edited away from the producer
text beside it must fail the verbatim-copy leg. A valid control confirms
that the faithful file still reads. (The chain-digest dataset identity
this leg once carried was ruled over-engineering and removed: which rows
trained is recorded by the staged-selection records, and which universe an
artifact belongs to is recorded by its facts — a byte digest of the chain
added only an identity layer on top of both.)

<a id="fixed-facts-schema-vertical-law-enforced"></a>
`fixed-facts-schema.vertical-law-enforced` provides a basic fixed-value check.
When the artifact and Cobaya's constant-parameter mapping expose a concrete
value under the same name, those values must agree. The error names both
values and the corrective action.

Missing, renamed, derived, and `n/a` values are left unchecked. Cobaya permits
arbitrary reparameterizations, so this name comparison cannot prove that two
cosmologies are equivalent. Compatibility of a custom parameterization remains
the user's responsibility.

<a id="fixed-facts-schema-horizontal-law-enforced"></a>
`fixed-facts-schema.horizontal-law-enforced` verifies that artifacts combined
in one prediction record the same fixed cosmology, the same conventions, and
the same sampled-coordinate set. A mismatch is refused with the disagreeing
fact and both values named. Two artifacts trained on different draws of the
same design pass: they approximate the same physical maps, so serving them
together is sound.

<a id="fixed-facts-schema-domain-law-enforced"></a>
`fixed-facts-schema.domain-law-enforced` verifies each requested point before
inference. Both support endpoints are accepted. A point outside the support is
refused with the interval, requested value, and corrective action. A record
with undeclared support is refused before numeric conversion, with the
synthetic generator named, rather than failing incidentally inside
`float("n/a")`.

<a id="fixed-facts-schema-served-support-is-the-intersection"></a>
`fixed-facts-schema.served-support-is-the-intersection` requires a pair's
served support to be the coordinate-wise intersection of both supports. A
point supported by only one artifact and a pair with a disjoint coordinate are
refused. Support therefore remains separate from the fixed-fact block that is
compared by equality.

<a id="fixed-facts-schema-comparison-laws-are-load-bearing"></a>
`fixed-facts-schema.comparison-laws-are-load-bearing` requires targeted
negative controls. The checks must fail when the fixed-facts comparison is
removed, undeclared support is accepted, an outside point is accepted, or
support union replaces intersection. Direct fixed-value match, mismatch,
missing-name, renamed-name, and `n/a` controls state the deliberately limited
runtime behavior. The unmodified implementation must pass every valid control.

<a id="fixed-facts-schema-resolved-model-read-once"></a>
`fixed-facts-schema.resolved-model-read-once` requires dataset generation to
call `fixed_facts.resolved_constants(model)`. The reader
uses this precedence: theory-component `extra_args` supply initial values; the
parameter block overrides duplicate names; and the first theory component
supplies names duplicated across components. It preserves the concrete names
Cobaya exposes and does not invent aliases. Boolean values remain Booleans,
and numeric values become floats. If the model cannot be inspected, unreadable
values remain absent and the record stores `n/a`. The check uses a small
model-shaped object and needs only NumPy.

<a id="cs-adapter-identity-adapter-contract"></a>
`cs-adapter-identity.adapter-contract` proves the cosmic-shear adapter reads its
configuration from the artifacts: the parameters it requires of the chain are the
emulator's own stored geometry names, and the vector it serves is the section the
stored geometry declares (`dv_return: 3x2pt` scatters into the full layout with
zeros off the mask). A wrong-kind artifact — a scalar emulator, which returns a
{name: value} dict rather than a vector — is refused by name, pointing at the
adapter it belongs in. This CPU-capable identity leg is required because the
separate `cobaya-adapter` integration gate needs CosmoLike and a GPU. Neither
gate may claim the other's capability boundary or evidence.

<a id="cs-adapter-identity-record-laws-refuse"></a>
`cs-adapter-identity.record-laws-refuse` requires the cosmic-shear adapter to
enforce all three comparison laws at their owning boundaries. After
configuration validation, initialization refuses artifacts that describe
different universes. When Cobaya supplies the provider, the adapter compares directly
named artifact constants with directly named model constants. An unavailable
or renamed value is inconclusive rather than a refusal. Before encoding each
point, `predict` refuses values outside stored support and records with
undeclared support. Each assertion checks law-specific error text so an
unrelated `ValueError`,
including one raised by `float("n/a")`, cannot satisfy the test.

<a id="artifact-readback-typed-bool"></a>
**Artifact readback parses saved attributes by type, not truthiness.** The
shared typed reader accepts a native Boolean and returns the declared default
for an absent optional key. The reader refuses every string or integer,
including the truthy string `"False"` that would otherwise load drifted
transfer weights. The refusal names the file and schema. A static search of
artifact-reading source confirms that
no Boolean field is coerced through truthiness. A real save, forged record,
and rebuild check requires the GPU-capable environment.

## Public returned arrays own their storage

**Reason.** On CPU, `.detach().cpu().numpy()` may share storage with a
persistent tensor because `cpu()` need not copy. On CUDA and the Apple Metal
Performance Shaders accelerator backend, transfer to CPU creates different
storage. Without
an explicit ownership rule, mutating a returned axis can therefore corrupt
later predictions on one device but not another. Persistent axes include
`self.z` and `self.k` in `emulator/inference.py` and `sigma`, `ell`, `scale`,
`z`, and `k` in public dictionaries from `emulator/diagnostics.py`.

**Rule.**

1. Every array a public entry point returns that derives from
   persistent model or geometry state is an owned copy. A caller
   mutation can never reach predictor or geometry state. Newly computed
   decoder results and model predictions have no
   second owner and are not blanket-copied. The contract is
   behavioral isolation, not defensive copying of everything.
2. A static search of `emulator/inference.py` and
   `emulator/diagnostics.py` starts from every `.numpy()` conversion and
   classifies each public return as persistent or newly computed.
3. Producer model mathematics remains unchanged. Any in-repository consumer
   that mutates a returned persistent array is reported as an ownership
   violation rather than accommodated silently.

**Acceptance evidence.** Call each affected public function, mutate every
returned array, call it again, and require both the stored axes and second
result to match an untouched reference exactly. Apply the same check to one
grid and one grid2d public diagnostic dictionary. Predictors built on CPU and
on the Apple accelerator backend must behave identically; CUDA uses the
GPU-capable environment. A mutation that
restores a storage-sharing `.numpy()` return must fail.

<a id="adapter-contracts-publication-and-owned-results"></a>
### Focused Cobaya publication and ownership evidence

`adapter-contracts.publication-and-owned-results` checks the five adapters'
public boundary. Scalar results must enter Cobaya's derived-result mapping;
CMB requests must use exact names and integer limits; matter-power artifacts
must use the correct quantity, units, and target law; and the covered Cobaya
getters must return owned arrays and containers. The focused checks do not
claim to cover every predictor or diagnostic return described below.

### Array ownership covers predictors, diagnostics, and Cobaya getters

**Reason.** A search limited to `.numpy()` misses adapters that return cached
calculation state directly. The affected public methods include
`emul_cmb.get_Cl`, `emul_cosmic_shear.get_cosmic_shear`, and
`emul_mps.get_Pk_grid`, whose result contains wavenumber, redshift, and power
arrays. A destructive first consumer could otherwise corrupt the provider
cache for every later consumer at the same sampled point.

**Rule.**

1. The ownership surface is every public exit: `EmulatorPredictor`
   returns, public diagnostic dictionaries, and every Cobaya getter
   across the five adapters.
2. The calculation state is the immutable cached scientific result;
   each public return owns its arrays and its mutable containers.
   The copy happens at the getter boundary — never by duplicating
   large arrays repeatedly inside calculate.
3. Nested structures are handled deliberately: the CMB dict plus its
   arrays; the MPS tuple plus all three arrays; the cosmic-shear
   vector. Immutable scalar returns need no copy.
4. Static analysis of the Python abstract syntax tree detects
   `return self.current_state[...]` and equivalent nested aliases; a text
   search for `.numpy()` alone is insufficient.
5. Docstrings state that "cache" means provider-owned and read-only
   to consumers, even though NumPy cannot enforce that read-only property.

**Acceptance evidence.** For each affected getter, read once, mutate every
returned array and mutable container, and read again. The second result and
`current_state` must match an untouched reference exactly. Two simulated
likelihood consumers exercise a destructive first reader and a read-only
second reader. MPS cases edit each of `k`, `z`, and `P`; CMB cases edit `ell`
and one spectrum. Mutations that restore each direct alias must fail. These
checks run on CPU.

## Model recipes are complete before construction

**Reason.** A model recipe is incomplete if any constructor field is absent,
even when the constructor supplies a default. Strict weight loading
cannot detect a missing parameterless activation because that activation has
no state-dictionary keys. Absence therefore differs from an explicit `None`:
for `head_act`, `None` means inherit the trunk activation, while absence means
artifact corruption.

**Rule.**

1. "Key absent" and "key present with explicit `None`" are distinct
   for every constructor field; absence raises before any
   import/construction.
2. The complete recipe validates before the model class is imported
   or constructed: required top-level keys, an exact kwargs schema
   for the declared class, complete factory specs, no unknown keys
   (unknown keys raise naming the class and the key).
3. `head_act` is required for a head model; explicit `None` is the valid
   "inherit the trunk activation" value; absence is corruption.
4. The required-field set derives from the class signature plus the documented
   injected fields `input_dim`, `output_dim`, `geom`, and factories. A new
   constructor default cannot reopen a fallback. The same rule governs
   `block_opts` and every optional lookup on the rebuild path.
5. Embedded transfer-base recipes validate under the same schema.
6. The live root model and live transfer base each expose their constructor
   recipe. Publication requires exact equality between those recipes
   and the corresponding claimed recipes.
7. Recipe dimensions agree with class-specific geometry facts before any
   model, geometry, activation, normalization, or Torch implementation is
   imported.
8. Complete artifacts stay byte-identical in prediction.

**Implementation boundary.** `emulator/model_recipe.py` owns the closed
six-class schema and performs the import-free check. `emulator/results.py`
calls that validator for the root recipe and each embedded transfer-base
recipe before construction or weight loading.

**Acceptance evidence.** Removing `head_act`, `block_opts.act`,
`block_opts.norm`, `n_blocks`, or another scalar constructor field one at a
time must cause refusal before construction. Explicit `head_act: null` must
preserve inherited behavior. Unknown keyword arguments name the class and
key. Embedded transfer-base recipes repeat the cases. Complete ordinary,
head, and transfer artifacts retain bitwise-identical predictions.

## Factored physical gain composes on the centered constant template

**Reason.** For the constant-coefficient template, the physical base is
`T0 + c`, where `c` is the geometry center. A multiplicative correction must
therefore compute `(T0 + c) * (1 + r0)`. Applying the correction to `T0` and
adding `c` afterward omits the cross term `c * r0`. For example, `c = 10`,
physical constant-template value `T0 + c = 12`, and `r0 = 1` produce the
incorrect value 14 instead of the
required value 24. A zero correction cannot expose this error, so parity at
`r0 = 0` is insufficient evidence.

**Rule.** The center is attached to the constant-coefficient template before
gain or sum composition and is not added again afterward. Frozen encoding and
chi-square calculation, `decode`, `base_decode`, and production inference
share one conversion and composition function. The constant-coefficient
template is identified by explicit IA metadata or a validated design rule,
not by an unexplained index. Sum, plain-transfer, whitened, and zero-correction
paths retain their established numerical behavior. An artifact trained with
an incompatible factored-gain formula is refused for retraining rather than
silently reinterpreted.

**Acceptance evidence.** The analytic example returns 24. A case with the
uncentered template value `T0 = 0`, so the physical constant-template value
equals `c`, retains nonzero gain leverage and gradient. The zero-correction,
factored-sum, plain-gain, and both whitened paths retain their expected
values. Frozen chi-square calculation, `decode`, and `base_decode` use the
same formula. A negative control that restores post-gain centering must fail.

## Public inference reads and enforces the `rescale` fact

**Reason.** `rescale` is required scientific metadata. Ignoring it installs
the plain decoder for an artifact that needs a parameter-dependent inverse
transformation. The result can be finite and correctly shaped yet wrong; the
recorded numerical example has a maximum absolute error of `28.236`.

**Rule.** A schema 3 writer publishes only an explicit native
`rescale: "none"` fact. If a caller also supplies a resolved value, that value
must be exactly the same; a missing, mistyped, transformed, or contradictory
fact refuses before temporary-file creation.

`rebuild_emulator` reads `rescale` as a required native string before model
execution. Public inference supports only `"none"`. Missing,
non-string, unknown, `"rescaled"`, and `"residual"` values are refused with an
explanation that the artifact does not contain enough information to rebuild
their inverse transformation. `EmulatorPredictor` and all five Cobaya adapters
use this shared check. Supporting a transformed form requires a new schema
that stores every decoder input, including `cosmo_mid`, `include_amp`,
`u_star`, and the theta and effective-redshift mapping, and then calls the
same training-loss decoder. The `"none"` path remains bitwise unchanged.

**Acceptance evidence.** Invalid types and values are refused before staging
on the write side and before model execution through the predictor and every
adapter on the read side. The `"none"` control is unchanged. A negative
control that removes the check reproduces the finite error of `28.236` and
must fail. Any future support for transformed inference requires separate
parameter-dependent checks for `"rescaled"` and `"residual"`. Each check must
use inputs for which the corresponding inverse transformation changes the
numerical result; one combined fixture is not enough.

## Artifacts record and enforce the physical parameter domain

**Reason.** `ParamGeometry.state()` records names and transformations but does
not by itself record where predictions are scientifically valid. A model
trained on `y = x` for `x` in `[-0.1, 0.1]` can return finite, correctly shaped
answers that are 23.84% wrong at `x = 1` and 90% wrong at `x = 10`. Finiteness
and type checks cannot detect this extrapolation.

**Rule.** Every artifact stores its admissible physical support by parameter
name. The support comes from the declared generator, prior, or cut, not from
observed sample extrema. A non-box constraint stores a named and versioned
validator rather than a widened bounding box. Save and rebuild validate names,
order, bounds, and the separation between sampled and fixed coordinates.
Where Cobaya exposes the sampler prior, adapter startup proves that prior is a
subset of the artifact support. Every requested point is also checked before
encoding. Values outside the support are refused and never clamped. When an
adapter combines artifacts, it serves only their declared intersection.
Fine-tuning and transfer may narrow inherited support but may not silently
widen it; every new coordinate receives explicit support. A legacy artifact
without this block is refused with a migration instruction. In-domain
predictions remain bitwise unchanged. NPCE uses this same domain record.

**Acceptance evidence.** A real save, rebuild, and scalar prediction covers
the support `[-0.1, 0.1]`. Both endpoints are accepted, while the nearest
representable values outside each endpoint and the finite `x = 1` and `x = 10`
examples are refused before encoding. A contained Cobaya prior is accepted and
a wider one is refused at startup. Tests cover overlapping and disjoint
multi-artifact domains, malformed and missing records, reordered names,
sampled-plus-fixed conflicts, and fine-tune and transfer propagation. A
negative control that removes the predictor check reproduces the 23.84% and
90% errors and must fail. `EmulatorPredictor` owns the shared enforcement.

## CosmoLike is imported only at the construction boundary that needs it

**Reason.** Only `DataVectorGeometry.from_cosmolike` uses
`cosmolike_lsst_y1_interface`. The plain constructor and `from_state` rebuild
path use persisted arrays. Requiring the compiled interface while importing
`emulator.geometries.output` would therefore prevent artifact-only operations
and Torch-only acceptance checks from reaching their own code.

**Rule.** The compiled interface is a dependency of the `from_cosmolike`
construction boundary, not of importing the persisted geometry type.
Importing `emulator.geometries.output`, constructing or restoring
`DataVectorGeometry` from explicit tensors, and rebuilding a saved artifact
must work with Torch and NumPy when the interface is absent. Calling
`from_cosmolike` without it must raise a teaching error that names the missing
compiled dependency and the operation that requested it. With the interface
installed, construction numerics and saved state remain unchanged.

**Acceptance evidence.** Execute the real `scalar-identity`,
`finetune-identity`, `transfer-identity`, and `finite-contract` child entry
points while deliberately making the interface unavailable. Each must pass
module import and reach its owned assertions. A mutation that restores an
eager module-level import must fail all four. These checks exercise production
geometry ownership. Test doubles that merely allow an import, without running
the production assertions, are not substitutes.

## Acceptance evidence: geometry module paths

The **gate registry** is the catalog in `ai/gates/board.py`, executed by
`ai/gates/run_board.py`. Its shared `repo_py_files()` function returns the
repository Python-file set used by whole-repository checks.

<a id="geo-paths-evidence"></a>
**geo-paths — fresh artifacts name geometry classes from the geometry package,
and the retired flat module paths remain absent.**

- files: creates a temporary covariance file plus temporary `.h5` and `.emul`
  artifact files; scans every repository Python file returned by
  `repo_py_files()` except its own check source, which contains
  the retired names as test data; all generated files are deleted with the
  temporary directory.
- subprocess: `ai/gates/checks/geo_paths.py`.
- metric: each named leg checks the expected geometry-class module prefix,
  exact attribute count, and finite prediction; the other legs perform a
  complete six-name disk/import census or a repository-Python reference
  census with the one named self-exclusion.
- legs: 3, named `geo-paths.fresh-save-uses-folder-paths`,
  `geo-paths.legacy-flat-paths-absent`, and
  `geo-paths.legacy-reference-census`.
- evidence: all three legs are asserted in the child; the child's exit status
  remains the single aggregate result and is not a fourth leg.
- capability boundary: the gate registry requires CPU PyTorch. NumPy and HDF5
  are ordinary child-process imports; if either is absent, the child command
  fails before these legs rather than reporting a capability skip. A pass of
  the complete registered acceptance suite is separate integration evidence
  and is not recorded as a `geo-paths` leg.

<a id="geo-paths-fresh-save-uses-folder-paths"></a>
`geo-paths.fresh-save-uses-folder-paths` requires a fresh artifact to contain
at least two attribute values beginning `emulator.geometries.`, no attribute
value beginning with the retired flat prefix, and a finite prediction after
rebuild. This evidence identifies the package prefix but does not identify
which two geometry classes own the markers; no class-specific claim may be
derived from this leg.

<a id="geo-paths-legacy-flat-paths-absent"></a>
`geo-paths.legacy-flat-paths-absent` checks each of the six retired module
names on disk and through `importlib.util.find_spec`; every name must be
absent.

<a id="geo-paths-legacy-reference-census"></a>
`geo-paths.legacy-reference-census` scans the complete repository Python-file
set supplied by `repo_py_files()`, excluding only the check that
contains the search terms, and requires zero retired flat-module references.

## Acceptance evidence: wrapper-family gates

The following blocks are the naming and evidence specification for six
artifact-lifecycle gates: two identity children that run in any CPU
environment with PyTorch; two smoke wrappers that read a real training
driver's output; and the paired save/rebuild and Cobaya integration checks,
which require CosmoLike and a GPU.

Three declared anchors require executable actions beyond the output checks
described by their wrappers. A log message or an instruction to inspect a file
is not acceptance evidence. Unless the named action executes, a wrapper must
record the anchor as unavailable or non-passing and must never infer a pass
from adjacent output.

<a id="finetune-identity-evidence"></a>
**finetune-identity — a warm-started emulator computes the source emulator's own
function before the first training step, whatever new parameters were added.**

- files: creates a temporary source artifact (`.h5` + `.emul`) and a temporary
  covariance file under a `ftw-` temporary directory; no CosmoLike installation
  or scientific dataset is required.
- subprocess: `ai/gates/checks/finetune_identity.py`.
- metric: exact tensor equality for the encoding, the transferred weights and
  the degenerate state dict; the parity leg reads the warm-start result line
  (`max|dv| = 0.000e+00` on 256 rows); every refusal case requires the declared
  exception and message.
- legs: 7, named `finetune-identity.extended-parameter-encoding`,
  `.weight-transfer-and-padding`, `.pre-train-parity`, `.output-geometry-pin`,
  `.degenerate-no-extras-identity`, `.loud-config-errors`, and
  `.anchor-mask-and-freedom`.
- evidence: all seven are asserted in the child, which emits one `##AID` per
  leg; the child's exit status stays the single aggregate result and is not an
  eighth leg.
- capability boundary: none. The child command passes in a CPU-only PyTorch
  environment.

<a id="finetune-identity-extended-parameter-encoding"></a>
`finetune-identity.extended-parameter-encoding` requires the extra names to be
`[w0, wa]` in covariance order, the shared coordinates to encode bit-identically
to the source, and the extra coordinates to be unmoved by a shared-only shift.

<a id="finetune-identity-weight-transfer-and-padding"></a>
`finetune-identity.weight-transfer-and-padding` requires the padded keys to be
exactly the input-consuming tensors, every unchanged tensor to be copied exactly,
and each padded tensor to be the source columns followed by exact zeros.

<a id="finetune-identity-pre-train-parity"></a>
`finetune-identity.pre-train-parity` requires `build_warm_start` to pass and
return its result line. `load_state_dict(init_state, strict=True)` must accept
the returned state dictionary without a missing or unexpected key.

<a id="finetune-identity-output-geometry-pin"></a>
`finetune-identity.output-geometry-pin` requires a matching dataset/probe/width
to reuse the source geometry object, and a data-vector width mismatch to raise.

<a id="finetune-identity-degenerate-no-extras-identity"></a>
`finetune-identity.degenerate-no-extras-identity` requires the no-extras case to
leave the geometry tensors and the transferred state dict exactly equal to the
source — the degenerate warm start is a copy.

<a id="finetune-identity-loud-config-errors"></a>
`finetune-identity.loud-config-errors` requires three raises: a non-superset
parameter set (naming the missing source parameter), a `model:` block beside
`finetune:`, and a `--rescale` other than `none`.

<a id="finetune-identity-anchor-mask-and-freedom"></a>
`finetune-identity.anchor-mask-and-freedom` requires the anchor mask to zero
exactly the padded extra columns, the source columns to be pinned to the
`init_state`, the padded extra columns to stay free, and the configured
`anchor: 0.0` case to be an exact no-op.

<a id="transfer-identity-evidence"></a>
**transfer-identity — a frozen base under a zero-output correction predicts the
frozen base itself, in every form and space, and a saved composition reloads to
the same prediction.**

- files: creates a temporary plain base, factored base, grid base and composed
  transfer artifact under a `tpe-` temporary directory; no CosmoLike
  installation or scientific dataset is required.
- subprocess: `ai/gates/checks/transfer_identity.py`.
- metric: exact tensor equality for the epoch-0 identity, the base-encoding
  slice and the save/rebuild composition; `1e-6` for the `EmulatorPredictor`
  comparison; call counting for the base cache; a raise for each refusal.
- legs: 8, named `transfer-identity.plain-base-slice-and-identity`,
  `.factored-base-slice-and-identity`, `.zero-init-surgery`,
  `.loud-config-errors`, `.artifact-lifecycle-round-trip`,
  `.refined-base-lifecycle`, `.diagonal-family-composition`, and
  `.cross-family-base-refusal`.
- evidence: all eight are asserted in the child, one `##AID` each. The two
  legs inside `check_diagonal` are emitted by that function rather than around
  it, so the cross-family refusal reports under its own name.
- capability boundary: none. The cross-family fixture uses a plain grid base
  so the family refusal, rather than the no-chaining refusal, is the only
  invalid condition.

<a id="transfer-identity-plain-base-slice-and-identity"></a>
`transfer-identity.plain-base-slice-and-identity` requires, for a plain base:
the extras to be `[w0, wa]`, the base encoding to be an exact column slice of
the run's encoding, and every combination in the two-by-two Cartesian product
of correction form and representation space to preserve the target width, the
base cache (one base encode and no chi-square recomputation), and the epoch-zero
identity with independence from the added coordinates.

<a id="transfer-identity-factored-base-slice-and-identity"></a>
`transfer-identity.factored-base-slice-and-identity` requires the same set for a
factored (three-template) base.

<a id="transfer-identity-zero-init-surgery"></a>
`transfer-identity.zero-init-surgery` requires the correction's final `Linear`
to be exactly zero (weight and bias) and every other tensor to be untouched.

<a id="transfer-identity-loud-config-errors"></a>
`transfer-identity.loud-config-errors` requires seven raises: an unknown
`transfer.form`; transfer with `--rescale`; transfer with pce; transfer with
finetune; transfer with `model.ia`; an incomplete `refine` block; and a
non-superset parameter set.

<a id="transfer-identity-artifact-lifecycle-round-trip"></a>
`transfer-identity.artifact-lifecycle-round-trip` requires a rebuilt transfer
artifact to return the embedded base with its form/space, its composed
prediction to equal the in-memory composition exactly,
`EmulatorPredictor.predict` to agree to `1e-6`, and chaining (a transfer used
as a base) to be refused.

<a id="transfer-identity-refined-base-lifecycle"></a>
`transfer-identity.refined-base-lifecycle` requires a refined artifact's
composed prediction to use the drifted base exactly, and a drifted state
without its companion attribute to raise.

<a id="transfer-identity-diagonal-family-composition"></a>
`transfer-identity.diagonal-family-composition` requires, on the diagonal
families: the epoch-0 identity through the log law for both forms, the packed
target with an exact zero-correction chi2, the refusal of physical space, the
transfer-validator resolutions and rejections, the family validators'
acceptance matrix, and a saved grid transfer artifact predicting the composition
exactly.

<a id="transfer-identity-cross-family-base-refusal"></a>
`transfer-identity.cross-family-base-refusal` requires `from_config` to raise a
`ValueError` naming the never-across-families rule when a grid2d run points at a
plain grid base. The fixture must violate only this rule. The leg retains its
own evidence identifier so a failure names the family refusal rather than the
larger composition group.

<a id="save-rebuild-drift-evidence"></a>
**save-rebuild-drift — an emulator rebuilt from its saved artifact pair reproduces
the live model exactly, its checkpoint is CPU-normalized, and a file the
schema cannot honour is refused.**

- files: trains and saves four tiny emulators under a `gsv-` temp directory, and
  persists one of them (the plain variant) to
  `<driver_root>/chains/gates_emul_evaluate` for the cobaya-adapter gate's
  evaluate leg to load. Here `<driver_root>` is the configured training-project
  directory. The child also reads the configured deployment data files.
- subprocess: `ai/gates/checks/gsv_bitwise_drift.py`.
- metric: exact tensor equality between the live and rebuilt outputs; exact CPU
  device type for every value in a nonempty tensor-only raw state dict; a raise
  (with the message named) for each refusal.
- legs: 9, named `save-rebuild-drift.plain-rebuild-matches-live`,
  `.cpu-normalized-state`, `.factored-rebuild-matches-live`,
  `.npce-rebuild-matches-live`, `.head-rebuild-matches-live`,
  `.code-default-drift-ignored`, `.v1-schema-refusal`,
  `.v2-schema-refusal`, and `.old-head-artifact-refusal`.
- evidence: all nine are asserted in the child, one `##AID` each; the wrapper's
  return-code check is the child's aggregate result and carries no separate
  `##AID` record.
- capability boundary: the whole gate needs CosmoLike, a CUDA GPU, and the
  configured deployment data files, so the gate is capability-skipped in a
  CPU-only environment. Only a CUDA-trained save can prove that CPU
  normalization moved a tensor rather than observing one that was already on
  the CPU. Execution therefore requires the GPU-capable environment.

<a id="save-rebuild-drift-plain-rebuild-matches-live"></a>
`save-rebuild-drift.plain-rebuild-matches-live` requires the plain variant's
rebuilt output to equal the live model's output exactly for the first eight
validation-input rows. In `gsv_bitwise_drift.py`, `exp` is the experiment
object, `val_set` is its validation-data mapping, and `"C"` selects the matrix
of cosmological parameter inputs.

<a id="save-rebuild-drift-cpu-normalized-state"></a>
`save-rebuild-drift.cpu-normalized-state` loads the just-written plain
checkpoint without `map_location` and requires a nonempty dictionary whose
values are all tensors and whose tensors all report `device.type == "cpu"`.
Without load-time relocation, the observed device is the one serialized by
`save_emulator`, not a destination selected by the check.

<a id="save-rebuild-drift-factored-rebuild-matches-live"></a>
`save-rebuild-drift.factored-rebuild-matches-live` requires the same for an
`nla` factored save.

<a id="save-rebuild-drift-npce-rebuild-matches-live"></a>
`save-rebuild-drift.npce-rebuild-matches-live` requires the same for a
neural-PCE save.

<a id="save-rebuild-drift-head-rebuild-matches-live"></a>
`save-rebuild-drift.head-rebuild-matches-live` requires the same for a
convolutional-head save. Rebuild must reconstruct the residual convolutional
network (`ResCNN`) from the persisted bin split alone, without reading a
dataset configuration file.

<a id="save-rebuild-drift-code-default-drift-ignored"></a>
`save-rebuild-drift.code-default-drift-ignored` monkeypatches
`make_activation`'s `n_gates` default (3 -> 7), rebuilds the plain save, and
requires the output to be unchanged: the rebuild reads the file, not the
activation code's runtime default. Compile-mode persistence is not claimed by
this arm because rebuild is deliberately called with `compile_model=False`.

<a id="save-rebuild-drift-v1-schema-refusal"></a>
`save-rebuild-drift.v1-schema-refusal` forges `schema_version` to 1 and requires
the rebuild to raise with the migration instruction named.

<a id="save-rebuild-drift-v2-schema-refusal"></a>
`save-rebuild-drift.v2-schema-refusal` forges `schema_version` to 2 and requires
the rebuild to raise with the migration instruction named. A v2 file carried no
record of the cosmology it was trained under, so it cannot prove it belongs to
the cosmology it is about to be asked about; the reader refuses it rather than
guessing. `emulator/fixed_facts.py` is the schema-version authority.

<a id="save-rebuild-drift-old-head-artifact-refusal"></a>
`save-rebuild-drift.old-head-artifact-refusal` deletes the persisted bin split
from a head save (a pre-persistence artifact) and requires the rebuild to raise
a `KeyError` naming the bin-split persistence — never to re-derive the split.

<a id="compile-recipe-evidence"></a>
**compile-recipe — a CUDA rebuild consumes the compile mode persisted in its
artifact.**

- files: writes opaque `case-a` and `case-b` schema-3 scalar artifact
  pairs under temporary directories, one with `compile_mode: default` and one
  with `compile_mode: reduce-overhead`. Paths and nearby descriptive labels do
  not encode either mode, and the check reads no configured deployment data.
- subprocess: `ai/gates/checks/compile_recipe.py`.
- metric: the independently read saved modes; each ordered `torch.compile`
  call's mode, uncompiled input callable, and returned compiled callable;
  explicit records of an identity compiler result or a rebuild that discards
  the compiler result; compiler or rebuilt-forward exceptions; and finite
  callable outputs.
- legs: 2, ordered as `compile-recipe.observation-controls` and
  `compile-recipe.cuda-persisted-modes`.
- evidence: the CPU leg writes and reads both real artifacts, exercises each
  rejected observation listed below, and forges a missing field through
  production rebuild. The CUDA leg records and delegates the real compiler
  call for both artifacts, then binds each compiled result to production's
  exact returned callable.
- capability boundary: the CPU leg does not require CUDA. The CUDA leg requires
  a CUDA-capable environment that supports both modes and executes both
  `compile_model=True` rebuilds. The
  standalone fixture needs neither CosmoLike nor configured deployment data.

<a id="compile-recipe-observation-controls"></a>
`compile-recipe.observation-controls` requires two real supported-schema saves to
persist the two distinct modes. Its ordered result accepts one matching call
per artifact and rejects a lost call, duplicate call, an exception from the
compiler or compiled forward pass, a swapped substitution, either hard-coded
mode, an identity compiler result, and a rebuild that discards the compiled
result. Deleting `compile_mode` from one
otherwise-valid recipe must make production `rebuild_emulator` raise a
`KeyError` naming the field and the no-code-default rule.

<a id="compile-recipe-cuda-persisted-modes"></a>
`compile-recipe.cuda-persisted-modes` first runs a tiny CUDA compiled forward in
both modes as the capability boundary. It then independently reads each saved
mode, rebuilds each artifact with `compile_model=True`, and records exactly one
matching call through a wrapper that delegates to the captured real
`torch.compile`. The delegated result must not be the eager input, production's
rebuild return must be that exact result, and it must produce a finite forward.
After both capability probes succeed, any save, read, rebuild, compile, or
forward exception is a test failure rather than a capability limitation. This
leg proves that the saved
value reaches the real call and production uses its returned callable; it does
not claim a particular internal PyTorch optimization strategy.

<a id="cobaya-adapter-evidence"></a>
**cobaya-adapter — the predictor a Cobaya theory block calls at sampling time
reproduces the training-side data vector and scatters it into the layout the
likelihood expects.**

- files: trains and saves two tiny emulators under a `gct-` temp directory;
  loads the emulator save-rebuild-drift persisted at
  `<driver_root>/chains/gates_emul_evaluate.h5`; runs the gate registry's
  evaluate YAML. The registry owns this generated dependency through
  `deps=("save-rebuild-drift",)` and runs that prerequisite first.
- subprocess: `ai/gates/checks/gct_parity.py` (the parity legs), then `cobaya-run`
  (the evaluate leg).
- metric: worst relative error <= `1e-6` for parity (denominator `|want| +
  1e-8`); set/length equality for the scattered-vector legs; process exit code
  for the evaluate run.
- legs: 7, named `cobaya-adapter.plain-predictor-parity`,
  `.plain-scattered-vector-shape-and-mask`, `.factored-predictor-parity`,
  `.factored-scattered-vector-shape-and-mask`, `.evaluate-emulator-present`,
  `.example-evaluate-run-completes`, and `.mcmc-smoke`.
- evidence: the four child parity legs emit one `##AID` each, and the wrapper
  executes the two evaluate legs. `mcmc-smoke` requires the separate short-chain
  action described below; evaluate output cannot satisfy it.
- capability boundary: CosmoLike, Cobaya, and a GPU are required. The wrapper
  must report `mcmc-smoke` as unavailable or non-passing unless it executes the
  short chain.

<a id="cobaya-adapter-plain-predictor-parity"></a>
`cobaya-adapter.plain-predictor-parity` requires the `EmulatorPredictor` built
from the saved plain file to match the training-side kept-entry data vector to a
worst relative error of `1e-6` across the first eight validation-input rows.
In `gct_parity.py`, `exp` is the experiment object, `val_set` is its
validation-data mapping, and `"C"` selects the matrix of cosmological parameter
inputs. Each row is paired with the parameter names saved in the artifact
geometry before predictor input is constructed.

<a id="cobaya-adapter-plain-scattered-vector-shape-and-mask"></a>
`cobaya-adapter.plain-scattered-vector-shape-and-mask` requires, for the plain
save, the section length to equal the stored `section_sizes[0]`, the 3x2pt
length to equal `total_size`, and every position outside `dest_idx` to be
exactly `0.0` in the scattered vector.

<a id="cobaya-adapter-factored-predictor-parity"></a>
`cobaya-adapter.factored-predictor-parity` requires the same parity bar for the
factored (`nla`) save, whose decode path runs through the chi2 function.

<a id="cobaya-adapter-factored-scattered-vector-shape-and-mask"></a>
`cobaya-adapter.factored-scattered-vector-shape-and-mask` requires the same
shape and masking set for the factored save.

<a id="cobaya-adapter-evaluate-emulator-present"></a>
`cobaya-adapter.evaluate-emulator-present` requires the emulator
produced by `save-rebuild-drift` to exist on disk before the evaluate run
starts. The gate registry automatically runs `save-rebuild-drift` first through
the declared `deps=("save-rebuild-drift",)` dependency, including when the
user requests only `--gate cobaya-adapter`. If the file is still absent after
that prerequisite, the gate fails before drawing any conclusion about Cobaya.

<a id="cobaya-adapter-example-evaluate-run-completes"></a>
`cobaya-adapter.example-evaluate-run-completes` requires `cobaya-run` on the
gate registry's evaluate YAML (the `lsst_y1` likelihood, `use_emulator 1`) to
exit zero.
It proves that the run completes. Numerical parity is established separately
by the named predictor assertion.

<a id="cobaya-adapter-mcmc-smoke"></a>
`cobaya-adapter.mcmc-smoke` requires a real short-chain run that drives the
theory block through a sampler, not merely through `evaluate`. A wrapper that
does not start the sampler must record this anchor as unavailable or
non-passing; it cannot use a successful evaluate run as substitute evidence.

<a id="finetune-smoke-evidence"></a>
**finetune-smoke — a real fine-tune run continues the gate registry's own saved
emulator.**

- files: reads the emulator produced by `save-rebuild-drift` under the
  configured driver file root; the run writes its own outputs under the driver
  root. The registry automatically supplies this file through
  `deps=("save-rebuild-drift",)`.
- subprocess: the cosmic-shear training driver, on the `finetune-smoke-config`
  YAML.
- metric: process exit code, plus exact presence of two required driver-output
  lines: the parity result and the warm-start banner.
- legs: 4, named `finetune-smoke.run-completes`, `.parity-verdict-printed`,
  `.warm-start-banner`, and `.artifact-provenance-and-round-trip`.
- evidence: the wrapper obtains the first three legs from the driver's exit
  code and output. The provenance/round-trip leg additionally requires the
  file action described below; output text cannot satisfy it.
- capability boundary: CosmoLike and a GPU are required.

<a id="finetune-smoke-run-completes"></a>
`finetune-smoke.run-completes` requires the fine-tune driver to exit zero.

<a id="finetune-smoke-parity-verdict-printed"></a>
`finetune-smoke.parity-verdict-printed` requires the driver's output to carry
the pre-train parity line (`finetune parity: max|dv|`). This is a text-presence
leg: it proves the driver ran the parity check and printed its result. The
identity itself is asserted numerically by finetune-identity.

<a id="finetune-smoke-warm-start-banner"></a>
`finetune-smoke.warm-start-banner` requires the startup banner to announce the
source artifact (`finetune: from `).

<a id="finetune-smoke-artifact-provenance-and-round-trip"></a>
`finetune-smoke.artifact-provenance-and-round-trip` requires the wrapper to open
the saved `.h5`, verify the `finetuned_from` root attribute, rebuild the
artifact with `rebuild_emulator`, and compare its prediction with the saved
run's reference prediction. A wrapper that inspects only standard output must
record this anchor as unavailable or non-passing. `finetune-identity` tests the
general mechanism but does not replace this file-specific evidence.

<a id="transfer-smoke-evidence"></a>
**transfer-smoke — a real transfer run composes a correction over the gate registry's own
saved base.**

- files: reads the plain base produced by `save-rebuild-drift` under the
  configured driver file root; the run saves its own composed artifact under
  the driver root. The registry automatically supplies this base through
  `deps=("save-rebuild-drift",)`.
- subprocess: the cosmic-shear training driver, on the `transfer-smoke-config`
  YAML.
- metric: process exit code, plus exact presence of four required driver-output
  lines: the epoch-zero parity result, the transfer banner, and the two save
  lines.
- legs: 5, named `transfer-smoke.run-completes`, `.parity-verdict-printed`,
  `.transfer-banner`, `.saved-artifact-paths-printed`, and
  `.artifact-provenance-and-round-trip`.
- evidence: the wrapper obtains the first four legs from the driver's exit code
  and output. The provenance/round-trip leg additionally requires the file
  action described below; output text cannot satisfy it.
- capability boundary: CosmoLike and a GPU are required.

<a id="transfer-smoke-run-completes"></a>
`transfer-smoke.run-completes` requires the transfer driver to exit zero.

<a id="transfer-smoke-parity-verdict-printed"></a>
`transfer-smoke.parity-verdict-printed` requires the driver's output to carry
the epoch-zero parity line (`transfer parity: epoch 0 == frozen base`). This
text-presence check proves that the driver ran and reported the parity check.
`transfer-identity` separately establishes the numerical identity.

<a id="transfer-smoke-transfer-banner"></a>
`transfer-smoke.transfer-banner` requires the startup banner to announce the
base and its form/space (`transfer: from `).

<a id="transfer-smoke-saved-artifact-paths-printed"></a>
`transfer-smoke.saved-artifact-paths-printed` requires both save lines (`saved
emulator ->` and `saved run record ->`) in the output. It proves the save ran
and printed its two paths. Reload behavior belongs to the next leg.

<a id="transfer-smoke-artifact-provenance-and-round-trip"></a>
`transfer-smoke.artifact-provenance-and-round-trip` requires the wrapper to
open the saved `.h5`, verify the `transfer_from` root attribute and embedded
`transfer_base` group, rebuild the artifact, and require the composed
prediction to reproduce the in-memory composition. A wrapper that inspects
only standard output must record this anchor as unavailable or non-passing.
`transfer-identity` tests the general lifecycle mechanism but does not replace
this file-specific evidence.
