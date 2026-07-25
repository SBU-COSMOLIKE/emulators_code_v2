# Artifacts, inference, adapters, and warm starts

Durable rules for saved emulators, reconstruction, public prediction, Cobaya
adapters, fine-tuning, and transfer learning.

**This note does not describe how the code works — the code does that.** Read
`emulator/results.py::save_emulator` (the write/read reversible map),
`::rebuild_emulator`, `emulator/inference.py`, `emulator/warmstart.py`,
`emulator/model_recipe.py`, `emulator/losses/transfer.py`,
`emulator/fixed_facts.py`, `emulator/geometries/`, and `cobaya_theory/`. Their
docstrings are the behavior specification and cannot drift from it.

What lives here instead, and nowhere else:

1. **Decisions and their reasons** — including designs considered and rejected,
   with the numbers that made the call. A docstring is the wrong home for "we
   tried X and refused it."
2. **Cross-file rules** no single function owns.
3. **Gate evidence specifications**, addressed by `<a id>` anchor from
   `ai/gates`. `run_board.py` refuses to start if an anchor does not resolve,
   so these are load-bearing text, not prose.

## Vocabulary

**Artifact** = one saved emulator: `<root>.emul` (torch checkpoint, weights,
CPU-normalized) + `<root>.h5` (HDF5, everything needed to interpret them).
**Pair token** = one fresh random string minted per save, written into both
members; a mismatched pair refuses, naming both tokens. Proves same-save
origin only — not a seal on bytes. **Schema** = versioned list and meaning of
saved fields. **Provenance** = where a thing came from; never proof it is
right.

Families: **CMB** (spectra `TT`/`TE`/`EE`), **BAOSN** (background expansion,
Hubble rate, transverse comoving distance `D_M`), **MPS** (matter power;
`pklin` linear, `boost` nonlinear/linear ratio), **IA** (intrinsic alignment,
`NLA` and `TATT`), **PCE** (polynomial chaos expansion) and **NPCE** (neural
correction over a frozen PCE base). `xi` = cosmic-shear correlation function,
`gammat` = tangential shear. `grid` = sampled along one coordinate; `grid2d` =
surface over redshift and wavenumber. **Syren** = the analytic linear-power or
boost baseline some MPS artifacts correct. `dv` = data vector, `chi2` =
covariance-weighted squared error. Whitened = rotated and scaled to identity
covariance; physical = original units.

Gate words: a **gate** is a named validation job whose required result is
written before it runs; a **leg** is one named assertion in it; the **child**
is the subprocess, the **wrapper** the harness; the child prints `##AID`
records binding results to evidence identifiers; `rc` is its return code. A
**control** must pass; a **mutation arm** weakens one rule and must make its
leg fail. **Capability-skipped** applies only when required software or
hardware is absent; **unavailable** means a declared action did not run and
therefore proves nothing. **Driver root** = the project directory the gate
configuration hands the training program; **driver file root** = the filename
stem; together, paths like `<driver_root>/chains/gates_emul_evaluate.h5`.

## Two identities, kept separate

Identity answers "same object?" Compatibility answers "may these be used
together?" Never conflate them.

1. **Staged-selection identity** — which source rows a run staged and in what
   order: source row count, split seed, physical cuts, selected count, and an
   order-sensitive digest of the exact disk rows; one record for training, one
   for validation. The dataset's own scientific description lives in the
   generator's `.facts.yaml`.
2. **Artifact identity** — resolved model and training recipes, output decoder
   and loss composition, staged-selection record, composition mode, and any
   source artifact or analytic base. A rebuilt emulator proves compatibility
   from the file alone.

Public inference does not need the training files, but must prove family,
product, parameter order, fixed facts, physical support, decoder, and
analytic-base implementation are compatible. **Matching shapes or filenames are
never compatibility evidence.**

## Save and rebuild: never reconstruct from code defaults

Defaults change while saved models stay in use. Write side materializes every
default at save time; read side reconstructs only from saved values and raises
loudly, naming the key and the migration action, rather than falling back;
display side renders saved values and never becomes a second source of truth.
Mechanics: `save_emulator` / `rebuild_emulator` docstrings.

Consequences worth stating once, because they are refusals rather than
behavior:

- Every successful save writes schema 3. No flag writes schema 1, 2, or a
  schemaless file; schema 1 and 2 reads refuse with a migration instruction. A
  v2 file cannot say which cosmology it was trained under, so it cannot prove
  it belongs to the one it is about to be asked about.
- An occupied root — complete pair, lone member, or symlink — refuses before
  any temporary file exists and stays byte-for-byte unchanged. A training run
  never replaces an existing emulator, because that would silently change what
  an existing chain's results meant.
- Both members write to temporary names and rename into place only when both
  are complete.
- `rebuild_emulator` loads with `weights_only=True` through a loader accepting
  only tensor values, so a rewritten checkpoint cannot execute pickle payloads
  or pass a non-tensor mapping off as a model state.
- A missing `.facts.yaml` is not permission to invent scientific metadata; the
  run stops with the generator's regeneration instruction.
- Geometry groups carry a `"cls"` attribute with the full module path, resolved
  through `importlib`. Rebuild never falls back to a base class: the file must
  identify the type, not only store numbers.
- Head artifacts rebuild from saved files alone. CosmoLike `DataVectorGeometry`
  persists its split (`bin_sizes`, and `pm_kept` when present) rather than
  rederiving it, because rederiving would need data files below `ROOTDIR`
  during inference. A head artifact without the persisted split refuses.
- Retired flat geometry module paths stay absent and raise
  `ModuleNotFoundError`; saves write folder paths through `type().__module__`.

**Run tags are labels, not identity.** `<model>[_t<T>]_ntrain<N>` is readable
shorthand; the scientific description is inside the artifact. Two runs the tag
cannot distinguish — CMB `TT` versus `EE` at one model and row count — need
different `--save` names, and the occupied-root refusal makes a collision
loud.

### Rejected: a second compatibility registry

A recipe records a class name and constructor values. It cannot prove the
Python behind a stable name never changed. A manually maintained
`model:...:v1` label would only repeat another name unless every scientific
change updated it correctly — so it would drift and be believed.

Rule: save concrete facts the prediction uses (model recipe, geometry state,
analytic-law name, composition mode, fixed facts, Git commit) and add no second
registry that copies them and claims to authenticate implementation behavior.
`git_commit` is provenance, not an execution lock: load an artifact with
edited local model code and checking that is the user's responsibility. If a
formula changes incompatibly, change its saved format or regenerate.

The check is that new root and transfer artifacts carry no duplicate
compatibility-manifest dataset while save-to-rebuild predictions stay
unchanged.

## Recipe completeness: absence is not a default

Strict weight loading cannot detect a missing parameterless activation, since
it has no state-dict keys. So **absent** and **present with explicit `None`**
must stay distinct for every constructor field. For `head_act`, `None` means
inherit the trunk activation; absence means corruption and raises before any
import or construction.

`emulator/model_recipe.py` closes over all six supported classes and validates
with plain non-executing values, importing no model, geometry, activation,
normalization, or Torch code — so saved text cannot select a Python class
before it is checked. Numerical limits stay with the constructor that uses the
value. The required-field set derives from the class signature plus the
injected `input_dim`, `output_dim`, `geom`, and factories, so a new constructor
default cannot reopen a fallback; the same governs `block_opts` and every
optional lookup on the rebuild path. Embedded transfer bases validate under the
same schema.

**The saved recipe must describe the live object.** Each constructor attaches
the canonical recipe for what it actually built, and `save_emulator` compares
it against the claimed root recipe (and the claimed embedded-base recipe for
transfer) before writing. A caller cannot save ordinary `ReLU` under a
registered gated-activation name, change the residual-layer count, or swap
classes while keeping a plausible dictionary. Geometry facts are checked
against the recipe and against each other: a self-consistent recipe cannot make
an inconsistent geometry safe.

`ai/tests/test_model_recipe.py` removes each scalar field in turn and owns
the case list. The census is the part that cannot be inferred: it compares the
registry against all six live constructor signatures and proves the validator
imports no executable model code, so adding a constructor field without its
saved representation must fail. Hard-coded duplicates (`compile_mode`) and
read-side keys no writer persists (`eval_bs`) are forbidden drift channels.

## Composition is declared, never inferred

Group presence cannot establish scientific composition. Delete an NPCE `pce`
group and the same-shaped weights strict-load under an ordinary decode and
return finite, different values: Hubble constant `69.68846130371094` becomes
`66.8885269165039`, matter-density fraction `0.31317755579948425` becomes
`0.30848246812820435`. Nothing looks broken. So weight-pair identity and
recipe-key totality do not replace an explicit composition fact.

Persist a native required enumeration — exactly one of `plain`, `npce`,
`transfer` — from the executed run, with transfer-refined state as a separate
native fact. Validate against the exact required and forbidden group set in
both directions before constructing anything. Schema-2 absence never means
plain; a presence-only artifact refuses with a migration instruction.
`config_yaml` may keep its `pce` or `transfer` block, but provenance YAML never
substitutes for the native fact — one authoritative consumed mode owns runtime
validation, and YAML is corroboration, not a second inference algorithm.

A census of every conditional HDF5 key found exactly one that silently
reinterprets a valid artifact when deleted: Grid2D `const_mask`, whose contract
lives in `families-background-mps.md`. Every other conditional key has an
explicit governing fact and refusal, raises before its consumer path, or is
covered by recipe and geometry totality. Cobaya adapters do no separate HDF5
presence dispatch.

<a id="artifact-composition-contract"></a>
**Artifact composition is authoritative before construction.** The writer
derives and persists native root facts `composition_mode` and
`transfer_refined` from the executed run; callers cannot replace either through
`attrs`. The resolved record carries top-level `composition_mode`,
`transfer_refined`, `pce`, and `transfer`.

Immediately after the scientific schema record is read, rebuild validates the
four legal rows — plain has neither base group; NPCE has only `pce`; frozen
transfer has only `transfer_base`; refined transfer additionally has
`transfer_base/drifted_state` — checking every required and forbidden edge in
both directions before recipe parsing, geometry or model construction, or
`torch.load`. Rebuild, inference, and warm-start then route on the validated
enumeration, never on optional-group presence.

The CPU/HDF5 gate proves the four valid rows, thirty focused forgeries,
writer/read agreement on the native form-and-space grammar, and the
pre-construction call order. The pair token proves only same-save origin; this
contract does not claim an attacker cannot rewrite every corroborating HDF5
surface together.

The discriminating negative check is a mutation restoring presence-only
inference: it must strict-load a same-shaped network and still fail the
known-answer prediction.

<a id="artifact-readback-typed-bool"></a>
**Artifact readback parses saved attributes by type, not truthiness.** The
shared typed reader accepts a native Boolean and returns the declared default
for an absent optional key, refusing every string or integer — including the
truthy string `"False"`, which would otherwise load drifted transfer weights.
Refusal names file and schema. A static search of artifact-reading source
confirms no Boolean field is coerced through truthiness. Real save, forged
record, and rebuild need the GPU-capable environment.

## Inference and the five adapters

`EmulatorPredictor` owns prediction physics. Every class in `cobaya_theory/` is
a thin adapter: no `nn.Module`, no duplicated physics. An MCMC YAML names
artifact path roots and never repeats architecture or whitening; geometry names
determine required parameters and `model_recipe` replaces the retired
`extrapar` convention, so neither fact has a second hand-maintained list.

Products: `emul_cosmic_shear` returns a data vector with
`dv_return: section|3x2pt` (`3x2pt` = the full combined three-probe layout;
default `section` lets the likelihood join per-probe sections using stored
`section_sizes` and `probe`); `emul_scalars` returns derived parameters;
`emul_cmb` implements `get_Cl`; `emul_baosn` gives piecewise Hubble and
distance products; `emul_mps` gives the power-spectrum grid and interpolator
used by the `EMUL2` theory component. Each refuses an incompatible artifact and
names the adapter that owns that kind.

**Load external adapters through `python_path`, not `path`.** Without it, an
incompatible adapter bundled in the CoCoA Cobaya fork can shadow the intended
class.

Transfer dispatch precedes intrinsic-alignment dispatch, or a factored transfer
correction enters the wrong branch. The MPS adapter, not the predictor,
restores the Syren analytic base. Geometry whitening tensors inherit float64;
Apple MPS does not support float64, so matter-power inference there may need an
explicit downcast. CPU and CUDA are the documented targets. For scripting
without Cobaya, see the README appendix "Scripting a saved emulator".

Two dependency contracts need **live Cobaya** coverage, because artifact-only
tests cannot establish routing:

- `emul_mps.get_can_support_params()` returns an empty list — that hook names
  sampled inputs a component can own, not calculated products. `get_Pk_grid`
  and `get_Pk_interpolator` advertise the products; `get_can_provide_params()`
  advertises `sigma8`. A gate that replaces Cobaya's `Theory` with a stub and
  hand-assigns `output_params = []` establishes nothing.
- With `want_derived` true, `emul_scalars.calculate()` creates
  `state["derived"]` when a direct caller supplied none, retains existing
  derived values, and publishes artifact outputs there. The optional `provides`
  list checks names and never filters that union. A real construction asks a
  likelihood for one scalar and requires it to travel the same route.

Requirement construction follows the artifact's parameter names and the Syren
alias rule below: an artifact naming `As_1e9` must not acquire a redundant `As`
requirement merely because it uses a Syren law. The EMUL2 example may define a
derived `As` bridge for another component; that bridge is not part of the
adapter's scientific requirement.

<a id="adapter-contracts-strict-inputs-and-composition"></a>

#### Focused input and cosmic-shear composition evidence

`adapter-contracts.strict-inputs-and-composition` checks the extra_args
refusals below, then builds concrete cosmic-shear section plans: disjoint
sections must follow physical block order, and overlaps, incompatible layouts,
repeated full vectors, and wrong widths must stop before becoming a likelihood
vector.

#### Adapter values, multi-emulator assembly, CMB requests

**Rejected: a shared strict-value validator module.** Each adapter checks its
own extra_args inline in a few direct lines. An unknown key is refused loudly,
naming accepted and retired keys. `emulators` must be a nonempty list — exactly
two entries for BAOSN or MPS. A relative root joins onto `ROOTDIR`. The device
pick resolves `cpu`/`cuda`/`mps` and falls back toward CPU when the requested
accelerator is missing. Everything else is read with the plain YAML types the
sampler produces. A central validation layer would add a subsystem where a
direct check suffices.

**Composition.** Never concatenate full-vector predictions blindly. Under
`dv_return: 3x2pt`, multiple predictors are refused unless one global vector is
assembled after proving compatible layouts and disjoint blocks. Section mode
requires compatible stored layouts and unique non-overlapping probe blocks.
Duplicate roots or probes, and a `3x2pt` artifact combined with a constituent
probe, fail before prediction; a valid disjoint multi-probe case keeps its
defined order. Blind `np.concatenate` would turn two full vectors into one
length-`2N` vector, or serve an overlapping likelihood block twice.

**CMB requests.** `must_provide` requires the `Cl` request to be a mapping,
every requested spectrum to be one a loaded artifact provides, and the
requested maximum multipole to sit inside the artifact's stored range. An
emulator has no accuracy beyond its training grid, so an out-of-range request
is refused — never truncated or zero-padded.

The MPS pair validator enforces the serving tuples
`pklin/Mpc3/(none|syren_linear)` and `boost/dimensionless/(none|syren_halofit)`;
the read side rejects a malformed or hand-built record rather than interpreting
an unsupported quantity-and-law combination as raw output.

The live-Cobaya cases are the ones artifact-only tests cannot reach:
dependency resolution assigns `Pk` products to the MPS theory, registers
`sigma8` as derived, calls the scalar adapter with `want_derived=True`, and
places advertised outputs in the returned state; an invalid MPS law-and-units
tuple and an `As_1e9`-only configuration with no redundant `As` bridge must
refuse.

## Numerical guards

### Geometry state and covariance

A class marker identifies a constructor; it does not prove the saved tensors
describe a finite, invertible transform. A zero scale, nonpositive eigenvalue,
malformed basis, duplicate destination index, or inconsistent dimension can
keep every weight shape valid while producing nonfinite values or the wrong
coordinate map. **Clipping a negative eigenvalue to zero only converts an
invalid covariance into a later division by zero** — reject it instead, naming
the smallest eigenvalue and the source or bin.

One shared validator runs in `ParamGeometry.from_covmat`, every sample- and
log-parameter builder, amplitude-factor and warm-start construction,
`DataVectorGeometry.from_cosmolike`, and every `from_state` rebuild — training
construction and HDF5 rebuild alike, before tensors reach a model. It checks
shapes, finiteness, unique in-range indices, monotonic finite axes, positive
scales and eigenvalues, orthonormal bases within a documented tolerance,
covariance symmetry and positive definiteness, and family registry/units
tuples. (`Cinv` = inverse covariance, `sqrt_ev` = covariance eigen-scales,
`dest_idx` = map from kept entries into the full vector.)

Gates cover a singular covariance block, a tiny negative eigenvalue, a zero
scale in a same-shaped h5, duplicate `dest_idx`, and a valid ill-conditioned
symmetric positive-definite matrix just above the tolerance.

### Parameter covariances share that contract

A one-parameter model is valid, but NumPy loads a scalar covariance file (`# x`
then `4.0`) as shape `()`, not `(1, 1)`, and `np.linalg.eigh` refuses that
dimensional accident; `np.cov(..., rowvar=False)` on one feature has the same
shape. A multi-parameter covariance with a negative variance instead reaches
`np.sqrt` and produces a NaN whitening scale. This is not a one-row
parameter-table shape error: the covariance is scientifically valid once its
one-dimensional representation is normalized.

So: normalize to an exact two-dimensional square matrix before
eigendecomposition — a valid scalar covariance becomes `(1, 1)` — while
normalization never rescues malformed input, and the normalized matrix still
passes every check. Header-name count, covariance width, and center width must
agree exactly, with all three observed dimensions named on mismatch. Require
finite, symmetric, strictly positive-definite values at every parameter-side
site, including `AmplitudeFactorGeometry.from_covmat` and the `output.py`
sites, and wherever covariance comes from samples, one feature included. Valid
multi-parameter results stay byte-for-byte identical.

Result: one-parameter emulators are buildable, and malformed covariances refuse
loudly instead of reaching training with a NaN whitening scale.

### Scales must survive the storage dtype

A relative check in float64 does not prove the stored float32 scale stays
positive: absolute underflow turns a valid-looking pre-cast scale into exact
zero, and encoding then divides by zero. `ScalarGeometry.from_targets`,
`GridGeometry.from_targets`, and `Grid2DGeometry.from_stats` all cross that
boundary.

Let `f = nextafter(float32(0), float32(1))`, about `1.4013e-45`. For targets
`[0, 0, f]` the float64 center is about `4.6710e-46` and the population scale
about `6.6058e-46` — a purely relative check accepts it, and both round to zero
in storage. Targets `[0, f, f]` keep a nonzero stored center but still round
the scale to zero. **The stored representation owns validity, not the pre-cast
ratio.**

Cast every center-and-scale pair through `float32` first, then require a finite
stored center and a finite strictly positive stored scale, naming the column or
grid coordinate on refusal. Apply the relative-resolution rule in that same
representation: `scale > 8 * float32_epsilon * abs(center)`. Both the absolute
underflow test and the relative-collapse test must pass. Grid2d classifies a
post-cast-zero scale as constant before deciding partial pin versus
whole-surface refusal. The same rule governs covariance square-root scales: a
positive float64 eigenvalue is insufficient if its stored float32 square root
is zero, and refusal names the smallest stored scale.

### Public inference validates inputs and outputs

Names and lengths do not validate a number. A Boolean, NaN, Inf, or nonscalar
can enter parameter whitening and propagate; decoding can produce a nonfinite
or wrong-shaped result that still looks structurally valid to an adapter.

`EmulatorPredictor._as_row` requires each supplied value to be a finite real
scalar and refuses Booleans, naming the stored parameter. The parameter
geometry must return finite encoded values. Model output and every decoder
branch must have the exact expected shape and finite values before conversion
to NumPy, a dict, or a scattered vector. `_as_row` owns input validation,
`CmbFactoredChi2._factor` the CMB domain, `EmulatorPredictor` the
post-encoding, post-model, and post-decoding checks shared by all adapters. A
refusal names which of those stages failed.

CMB adds a family domain: `as_exp2tau_ref` divides a stored reference amplitude
by the sampled amplitude and exponentiates twice the optical-depth difference,
so `A_s` must be strictly positive and `tau` finite, and the factor itself must
be strictly positive and finite. BAOSN and MPS keep their own positivity rules;
none applies to `TE` or a generic scalar output. The complete law, including
refusal of its retired raw form, is owned by `families-scalar-cmb.md`.

### Public arrays own their storage

On CPU, `.detach().cpu().numpy()` may share storage with a persistent tensor,
because `cpu()` need not copy. On CUDA and Apple MPS the transfer creates new
storage. Without an explicit rule, mutating a returned axis corrupts later
predictions **on one device but not another**. Persistent axes include `self.z`
and `self.k` in `inference.py` and `sigma`, `ell`, `scale`, `z`, `k` in public
dictionaries from `diagnostics.py`.

Rule: every array a public entry point returns that derives from persistent
model or geometry state is an owned copy, so a caller mutation can never reach
predictor or geometry state. Newly computed decoder results and model
predictions have no second owner and are not blanket-copied — the contract is
behavioral isolation, not defensive copying of everything. Producer mathematics
is unchanged, and an in-repository consumer that mutates a returned persistent
array is reported as a violation rather than accommodated.

A search limited to `.numpy()` is insufficient: adapters return cached
calculation state directly (`emul_cmb.get_Cl`,
`emul_cosmic_shear.get_cosmic_shear`, `emul_mps.get_Pk_grid` with its
wavenumber, redshift, and power arrays). A destructive first consumer would
corrupt the provider cache for every later consumer at the same sampled point.
So the ownership surface is every public exit — predictor returns, public
diagnostic dictionaries, and every Cobaya getter across the five adapters — and
the copy happens at the getter boundary, never by duplicating large arrays
repeatedly inside `calculate`. Nested structures are handled deliberately: the
CMB dict and its arrays, the MPS tuple and all three arrays, the cosmic-shear
vector; immutable scalars need no copy. Static analysis of the syntax tree
detects `return self.current_state[...]` and equivalent nested aliases.
Docstrings state that "cache" means provider-owned and read-only to consumers,
even though NumPy cannot enforce it.

The discriminating check is read once, mutate every returned array and mutable
container, read again: the second result and `current_state` must match an
untouched reference exactly. Mutations restoring each direct alias, or a
storage-sharing `.numpy()` return, must fail, and CPU and Apple-backend
predictors must behave identically.

<a id="adapter-contracts-publication-and-owned-results"></a>
`adapter-contracts.publication-and-owned-results` checks the five adapters'
public boundary: scalar results must enter Cobaya's derived-result mapping, CMB
requests must use exact names and integer limits, matter-power artifacts must
use the correct quantity, units, and target law, and the covered getters must
return owned arrays and containers. It does not claim to cover every predictor
or diagnostic return above.

## Serving domain and the `rescale` fact

### `rescale`

Ignoring `rescale` installs the plain decoder for an artifact needing a
parameter-dependent inverse transform. The result is finite, correctly shaped,
and wrong — recorded maximum absolute error `28.236`.

A schema 3 writer publishes only an explicit native `rescale: "none"`. A
caller-supplied resolved value must match exactly; missing, mistyped,
transformed, or contradictory refuses before temporary-file creation.
`rebuild_emulator` reads it as a required native string before model execution,
and public inference supports only `"none"` — missing, non-string, unknown,
`"rescaled"`, and `"residual"` refuse with the explanation that the artifact
does not carry enough information to rebuild the inverse transform.
`EmulatorPredictor` and all five adapters share that check. Supporting a
transformed form requires a new schema storing every decoder input —
`cosmo_mid`, `include_amp`, `u_star`, and the theta and effective-redshift
mapping — and then calling the same training-loss decoder. The `"none"` path
stays bitwise unchanged.

Evidence: invalid types and values refuse before staging on the write side and
before model execution through the predictor and every adapter on the read
side; the `"none"` control is unchanged; a negative control removing the check
reproduces the `28.236` error and must fail. Future transformed support needs
separate checks for `"rescaled"` and `"residual"`, each with inputs where the
inverse transform changes the number — one combined fixture is not enough.

### Physical parameter domain

`ParamGeometry.state()` records names and transformations, not where
predictions are valid. A model trained on `y = x` over `[-0.1, 0.1]` returns
finite, correctly shaped answers that are **23.84% wrong at `x = 1` and 90%
wrong at `x = 10`**. Finiteness and type checks cannot see this.

Every artifact stores admissible physical support by parameter name, taken from
the declared generator, prior, or cut — never from observed sample extrema. A
non-box constraint stores a named, versioned validator rather than a widened
bounding box. Save and rebuild validate names, order, bounds, and the
separation of sampled from fixed coordinates. Where Cobaya exposes the sampler
prior, adapter startup proves it is a subset of the artifact support. Every
requested point is checked before encoding, and values outside support are
refused, never clamped. Combining artifacts serves only their declared
intersection. Fine-tuning and transfer may narrow inherited support but never
silently widen it, and every new coordinate gets explicit support. A legacy
artifact without this block refuses with a migration instruction. In-domain
predictions stay bitwise unchanged; NPCE uses the same record.

Evidence: a real save, rebuild, and scalar prediction over `[-0.1, 0.1]`; both
endpoints accepted; the nearest representable values outside each endpoint and
the finite `x = 1` and `x = 10` cases refused before encoding; a contained
Cobaya prior accepted and a wider one refused at startup; overlapping and
disjoint multi-artifact domains, malformed and missing records, reordered
names, sampled-plus-fixed conflicts, and fine-tune and transfer propagation.
A negative control removing the predictor check reproduces 23.84% and 90% and
must fail. `EmulatorPredictor` owns the shared enforcement.

### Factored physical gain composes on the centered template

For the constant-coefficient template the physical base is `T0 + c`, where `c`
is the geometry center, so a multiplicative correction computes
`(T0 + c) * (1 + r0)`. Correcting `T0` and adding `c` afterwards drops the
cross term `c * r0`: with `c = 10`, `T0 + c = 12`, `r0 = 1`, the wrong route
gives 14 instead of 24. **A zero correction cannot expose this, so parity at
`r0 = 0` is not sufficient evidence.**

The center attaches to the constant-coefficient template before gain or sum
composition and is never added again. Frozen encoding and chi-square,
`decode`, `base_decode`, and production inference share one conversion and
composition function. The template is identified by explicit IA metadata or a
validated design rule, never an unexplained index. Sum, plain-transfer,
whitened, and zero-correction paths keep their established numerics. An
artifact trained with an incompatible factored-gain formula is refused for
retraining rather than silently reinterpreted.

The analytic example must return 24, and a negative control restoring
post-gain centering must fail. A case with uncentered `T0 = 0`, where the
physical template value equals `c`, must keep nonzero gain leverage and
gradient.

## Fine-tuning

`train_args.finetune` takes `from` and optional `compile_mode`. The source HDF5
record owns the architecture; a sibling `model` block is refused. A lower
learning rate goes through the ordinary `lr` block — one decade below the
source with `warmup_epochs >= 3` is the teaching recommendation, because the
optimizer moments start cold. Provenance: `finetuned_from` and
`finetune_extra_names`. Mechanics: `emulator/warmstart.py`.

**Epoch-zero rule.** Before the first update the warm-started model computes
the source function independently of the added parameters: `max|dv| <= 1e-5` in
float32, exact zero for equal-name runs.

The parameter geometry extends in blocks, layout `[shared; extras; raw
amplitudes]`: shared rows keep the source rotation bit for bit, added
coordinates use their marginal covariance block, and grown input columns follow
the source columns with exact zeros. **Cross-correlations between added and
shared inputs are deliberately not whitened**, because that would change the
shared encoding. The source artifact pins the output-geometry class and state,
and a wrong family or mismatched metadata refuses.

Scalar fine-tuning needs the same source provenance as every other family: a
save path recording only model, data, and best-metric attributes cannot
distinguish a cold run from one whose architecture and initial weights came
from another artifact. One shared provenance assembler owns the common
attributes for every driver — scalar adds its family facts and never forks the
fine-tuning, anchor, or source-provenance logic. A cold run stores no fine-tune
attributes; a fine-tune run stores the canonical resolved source identity (the
root and digest actually loaded, never the raw YAML spelling) and the ordered
extra names. A mutation removing the shared provenance call must fail artifact
readback.

### Warm-start source reads and perturbed values

`FinetuneSource` is one in-memory object, and a successful construction opens
the source HDF5 twice: `rebuild_emulator` owns the first open and loads the
`.emul` weights once; `load_source` owns the second, because the warm-start
validator needs the model recipe, saved rescale value, and resolved data block
that `rebuild_emulator` does not return.

Both parity paths name and screen the two values produced only after the
extra-coordinate perturbation — fine-tuning screens `enc_pert` then `out_pert`,
transfer screens `enc_pert` then `composed_pert`. All four use
`_require_parity_finite`, whose shared error names the pipeline side, the
quantity, and the staged source-row coordinates, and the comparison runs only
after both values are finite. Screening order is the point: skipping an input
guard shifts the reported quantity to the later output, and skipping an output
guard restores the misleading `extra parameters leaked` or `extra parameters
moved` diagnosis. `ai/tests/test_warmstart_perturbed_finite.py` owns the cases,
and each mutation test first verifies that the targeted production guard
executes. `finite_contract.py` Parts D and E carry the gate legs; they count
only when the registered gate executes them.

### The anchor

**Availability boundary.** `train_args.finetune.anchor` is refused until the
complete contract here is implemented and its registered gate passes;
`validate_finetune_config` raises for the key rather than advertising an update
that may match no parameters.

Reason: source reference tensors and masks use eager-module parameter names,
while a compiled live model can expose `_orig_mod.`-prefixed names. If
`build_anchor` merely skipped absent reference names, a positive anchor could
produce zero entries and no update while the artifact still recorded the
configured value.

Enabling requires: a finite nonnegative real strength (NaN and either infinity
refused); compile prefixes canonicalized at one boundary, or anchoring on the
underlying eager module, with exact one-to-one parameter coverage; refusal of a
positive strength matching zero trainable parameters; reporting and validation
of matched, masked, frozen, and unexpected names; masks under the same
canonical mapping; and an artifact recording *executed* anchor evidence —
matched count and effective strength — not configuration alone.

**Update.** With `W` a current trainable parameter, `W_0` its saved source,
`lr` the learning rate, `lambda` the configured strength, and `mask` the binary
selection: `W <- W - lr*lambda*mask*(W - W_0)`, applied after the optimizer
step. Added input columns carry the new physics and get mask zero. **The anchor
is deliberately not part of the scalar loss**, because Adam would rescale it
through its stored moments. Pair a nonzero anchor with `weight_decay: 0.0`.

#### What the README must teach about the anchor

The root README explains the anchor as weight-displacement regularization, not
as a loss term. "A pull back toward the saved weights" is not sufficient alone.
It must define L2 starting-point regularization (`L2-SP`) with every symbol
defined, state that ordinary weight decay pulls toward zero while this measures
movement away from an already-trained source, and then give the executable
truth: the library does **not** add the penalty to the scalar loss AdamW sees.
After the ordinary step proposes `W_j^opt`, `Anchor.apply` performs a decoupled
in-place update using that parameter group's current learning rate, which keeps
Adam's second-moment rescaling out of the anchor. There is no hidden division
by batch size, layer width, or parameter count, and `anchor: 0.0` is an exact
no-op. One worked number is required, and the text must say the optimizer's
scientific step happened first.

On selection the README must say that matched trainable source parameters are
anchored whenever an optimizer group owns them, that geometry tensors and other
non-parameter state are not, and that input columns added by new cosmological
inputs start at zero for epoch-0 parity and carry mask zero so they stay free
while pre-existing columns carry mask one. On availability it must say that
ordinary fine-tuning is unanchored while `validate_finetune_config` refuses the
key, that no usable fine-tune-anchor YAML may be printed unless the registered
gate passes, and that the separate cosmic-shear `transfer.refine.anchor`
anchors only the frozen base during joint refinement, never the correction
network, with diagonal-family transfer refinement refused.

`emulator/README.md` carries a shorter equation-and-owner pointer, not the full
tutorial. The prose must never say the implementation adds an L2 term "to the
loss," and a claim-consistency scan covers "anchor," "L2-SP," "penalty," and
"refine" across both READMEs.

## Transfer learning (all families except scalar)

Scalar transfer is unsupported as an explicit product boundary, not a
structural limit. Transfer is judged by **sample efficiency** — accuracy per
expensive training cosmology — not wall-clock time.

The trained base freezes as a whole; a small parallel correction network
receives the complete new parameter space and produces `r`. Form `gain` gives
`base * (1 + r)`, form `sum` gives `base + r`, composed in physical or whitened
space. An omitted space resolves to the form's documented recommendation and is
saved as an explicit value.

Diagonal families use `TransferDiagChi2` (a `CmbDiagonalChi2` subclass) and
accept plain bases and whitened space only, because physical composition is
separated from their metric by an elementwise scale and can cross a logarithmic
law domain. Both forms stay available, with a zero-crossing notice for `gain`
and `sum` recommended. `transfer.refine` and roughness with transfer are
refused. CMB requires `amplitude_law: none` on both sides so one
target-construction rule owns the data. Compatibility pins spectrum, multipole
coordinates, scales, redshift coordinates, quantity, units, and law as
applicable; a cross-family base is refused by `from_config`.

**Efficiency.** The frozen base runs once per row during encoding; the staged
target stores `[base; truth]` and repeated chi-square composes those cached
values without re-evaluating the base. Hook counts establish the
one-evaluation property.

**Identity.** An exactly zero correction reproduces the frozen-base decode
bitwise when both sides use the same arithmetic path. The factored physical
path reassociates template combination and unwhitening, so it uses a documented
`1e-6` to `1e-5` tolerance; measured reference difference about `4e-6`.

**Artifact.** `transfer_base` embeds the base recipe, state, both geometries,
form, and space — never an external reference — and chaining is refused.
Optional stage-two `transfer.refine` unfreezes the base once, applies per-group
`base_lr_scale`, and requires an explicit anchor strength including `0.0`. A
refined artifact keeps pretrained reference weights in `transfer_base` and
prediction weights in `drifted_state`, and the two states must permit exact
drift recomputation. Rebuilding selects `state` or `drifted_state` from the
explicit refinement fact and loads it strictly, so missing, unexpected, or
wrong-shaped tensors refuse. The file does not hash the embedded mapping again
or copy that hash into configuration records — a same-shaped value edit inside
the HDF5 is user responsibility. The resolved record names the source path
root, form, and materialized space.

The four supported training modes are from-scratch, anchored warm start,
frozen-base transfer, and anchored joint refinement; the decoupled L2-SP
strength spans the refinement range from frozen to free.

### Refusal fixtures violate exactly one rule

A refusal fixture must violate only the rule under test. The cross-family
transfer leg therefore saves a plain `GridGeometry` base and points a grid2d
configuration at it: a *transfer* artifact would violate both the no-chaining
and cross-family rules, letting the no-chaining error fire first and hiding
whether the family check works. `ai/gates/checks/transfer_identity.py` owns the
fixture; the production family check stays in `_load_diag_transfer`, and
production code must not be changed to accommodate an ambiguous fixture. CMB
identity metadata must describe all five covariance checks in plain language,
and a GPU environment must rerun both identity gates after a fixture change.

### Transfer-refine drift measures trainable parameters only

A relative weight-drift metric must exclude persistent non-trainable buffers —
`pad_idx` layout indices from `ResCNN` and `ResTRF`, fixed PCE buffers — which
contribute zero to the numerator but inflate the denominator and dilute the
reported change. Relative drift is also undefined when the reference norm is
zero: **a moved zero tensor must never be reported as relative drift `0.0`.**

Define the metric over trainable parameters through an explicit canonical key
set, excluding buffers, layout, and other state. Persist the numerator and
reference norms, or absolute drift plus a named status, beside any relative
value. If `||W0|| == 0`, report exact zero only when the drift norm is also
zero; otherwise report absolute drift with a `zero-reference` status. Verify
parameter-key equality between the two states before saving the summary, and
require the declared key set plus the two persisted states to reproduce the
stored summary exactly. If the metric includes non-parameter state, call it
state drift, not weight drift.

## `config_resolved_yaml` records what the run consumed

Raw inputs do not describe consumed values: an absent loss block consumes the
default square-root mode, omitted BerHu knots consume materialized numbers, and
each phase resolves effective loss, knots, EMA, trimming, focus, clipping,
rewind, learning rate, warmup, and scheduler values. Transfer refinement
inherits another effective pass. A record holding only raw trunk/head overlays,
one top-level scheduler, or `{epochs, base_lr_scale, anchor}` would force a
reader to rerun default and inheritance logic, and history rows can include
refinement while a pre-refinement epoch count does not.

Persist a `passes` sequence in execution order, each entry recording phase
name, model training phase, epoch count, computed learning rate, warmup,
scheduler class and resolved keyword arguments, fully resolved loss, trimming,
focus, clipping, rewind, and EMA. Transfer refinement gets its own pass entry
with every inherited effective value. Persist run-level roughness separately
when one value configures the loss object for the whole run, plus
`total_epochs` and the five history arrays. Keep raw YAML separately as
provenance.

**Prediction does not reconstruct training.** The artifact writer checks only
that history arrays are finite and compatibly shaped; reopening treats the pass
plan and history arrays as provenance, and reconstruction never reads the
`history` group or validates pass grammar. `run_emulator` owns the complete
pass records, and the writer must not become a second training-policy engine.
Removing the entire history group from a valid artifact must still rebuild,
which is what proves historical curves are not prediction inputs.

A two-phase run with two trunk and three head epochs records contiguous slices
`[0, 2)` and `[2, 5)` and `total_epochs: 5`; four saved loss rows cannot
describe that run and must refuse. A refinement pass appears after the ordinary
or trunk/head passes and extends the same contiguous history rather than
starting a new counter. `ai/tests/test_training_pass_recipe.py` and
`ai/tests/test_artifact_recipe_preflight.py` own the cases.

## Syren parameter aliases must agree

The Syren formulas read two equivalent amplitude names, `As_1e9` and `As`, and
several equivalent dark-energy forms (`w` and `w0` name the present-day
equation-of-state value, `wa` its time evolution, `w0pwa` means `w0 + wa`).
Silently choosing one repeated value makes the network and the analytic
starting formula describe **different cosmologies**, and the failure stays
finite, so downstream checks miss it:

- `{As_1e9: 2.1, As: 9e-9}` — choosing `As_1e9` shifts the analytic
  linear-power baseline by a maximum relative difference of `0.7667`.
- `{w: -1.0, w0: -0.7}` — choosing `w` shifts it by a grid-dependent maximum
  near `0.2449`.

A law-correction artifact is especially exposed: the network reads the
artifact's stored parameter names while the analytic base could follow a
separate alias-precedence rule.

One input path:

1. `resolve_dark_energy_coordinates` in `emulator/syren_base.py` is the shared
   authority for `w`, `w0`, `wa`, `w0pwa`. `syren_params_from` is the one entry
   point for the seven Syren arguments and delegates the last two to that
   resolver. Complete law and completion rules: `families-background-mps.md`.
2. `As_1e9` alone and `As` alone are accepted. Both present requires
   `As_1e9 == 1e9 * As` within a documented float-representation tolerance;
   otherwise raise naming both values and the conversion.
3. `w` and `w0` are two names for one value. Either may be used when the input
   also supplies `wa` or `w0pwa`, or when the explicit saved law supplies the
   missing coordinate. Both present must agree, and a mismatch names both.
4. A complete transformed input supplies a present-day alias and `w0pwa`, and
   the resolver derives `wa`; if `wa` is also present, supplied and derived
   must agree.
5. Incomplete coordinates never select a law; each saved law's completion
   values are owned by `families-background-mps.md`, and every repeated value
   is checked against them.
6. Canonicalize only after every repeated value agrees. Never prefer one
   conflicting alias, never replace missing evolution information with zero.
7. Do not duplicate the individual-value contract: the generator validates
   values entering generation and the adapter validates values from Cobaya.
   This rule owns relationships among aliases and transformed coordinates.
8. Requirement construction requests no redundant amplitude alias. For
   time-varying dark energy the adapter requests the present-day alias and
   calculated `wa`, not dropped `w0pwa`, then rebuilds every saved spelling
   before prediction.
9. On failure `emul_mps.calculate` leaves no `Pk_grid`, interpolator, or
   derived state key, and the generator refuses the sample before writing a raw
   or starting-surface row.
10. Documentation defines `As_1e9 = 10^9 As`, defines `w` and `w0` as two names
    for one value, and defines `w0pwa = w0 + wa`. It never describes
    correctness as one alias being "preferred."

Acceptance cases must use the real routed inputs, not hand-built mappings: the
generator receives Cobaya's complete calculated input mapping, while the MPS
adapter asks Cobaya for the saved present-day name and calculated `wa` and then
rebuilds all four names before prediction. The shipped EMUL2 evaluation
configuration supplies both amplitude names, so dual-amplitude input is a
public configuration shape. The real-Cobaya check sends `w = -0.9, wa = 0.2`
from sampled `w = -0.9, w0pwa = -0.7`, and generation and serving must give
Syren the same pair. `ai/tests/test_mps_amplitude_aliases.py` owns the
amplitude cases.

## CosmoLike imports only at the boundary that needs it

Only `DataVectorGeometry.from_cosmolike` uses `cosmolike_lsst_y1_interface`;
the plain constructor and `from_state` use persisted arrays. Requiring the
compiled interface to import `emulator.geometries.output` would block
artifact-only operations and Torch-only acceptance checks from reaching their
own code.

So the compiled interface is a dependency of the `from_cosmolike` construction
boundary, not of importing the persisted type. Importing the module,
constructing or restoring `DataVectorGeometry` from explicit tensors, and
rebuilding a saved artifact must work with Torch and NumPy alone. Calling
`from_cosmolike` without it raises a teaching error naming the missing compiled
dependency and the operation that requested it.

The proof is running the real `scalar-identity`, `finetune-identity`,
`transfer-identity`, and `finite-contract` child entry points with the
interface deliberately unavailable: each must pass module import and reach its
owned assertions, and a mutation restoring an eager module-level import must
fail all four. Test doubles that merely allow an import, without running the
production assertions, are not substitutes.

## Code ownership map

- Save/rebuild and schema: `emulator/results.py`, `emulator/fixed_facts.py`
- Model recipe: `emulator/model_recipe.py`
- Ordered training-pass construction: `emulator/training.py::run_emulator`
- Save-time history shape checks: `results.py::_history_arrays_for_save`
- Public prediction: `emulator/inference.py`
- Fine-tune and transfer source: `emulator/warmstart.py`
- Transfer composition: `emulator/losses/transfer.py`
- Adapters: `cobaya_theory/`
- Registered gates: `save-rebuild-drift`, `cobaya-adapter`, `finetune-identity`,
  `finetune-smoke`, `transfer-identity`, `transfer-smoke`, `geo-paths`

## Structured acceptance evidence

Each gate links every named assertion to one `<a id>` anchor in this note. The
permanent-note evidence validator refuses a missing anchor or a repeated
evidence identifier.

`ai/gates/board.py` is the registry: each `Gate(...)` entry owns the manifest,
the subprocess or child, the leg names, and the required capabilities, so the
blocks below do not repeat them. What each block records instead is **metric**
(the comparison or refusal that decides success), **evidence** (how the output
proves each leg), and **capability boundary** (what a missing capability means
— never a passing result inferred from adjacent output).

### fixed-facts-schema: the science an emulator was born under

`emulator/fixed_facts.py` stores two sibling blocks in a real HDF5 file.
`fixed_facts` records coordinates held constant while sampled ones varied, and
is compared by equality. `input_domain` records sampled support, and is
compared by overlap. Keeping them separate prevents per-key exceptions to one
ambiguous comparison rule. No accelerator needed.

<a id="fixed-facts-schema-record-round-trip"></a>
`fixed-facts-schema.record-round-trip` writes both blocks to a real file and
reads back: sampled names survive in order, a boolean fact returns a boolean
(HDF5 has no Python types, and `True == 1` in Python), the fixed cosmology
survives, a sampled coordinate is absent from the fixed cosmology, and two
bounds differing in the last float32 digit stay distinct under the shortest
round-tripping decimal.

<a id="fixed-facts-schema-rewritten-record-refused"></a>
`fixed-facts-schema.rewritten-record-refused` proves the two-way check behind
"copied verbatim, never re-derived": the file carries the producer's own text
and the blocks parsed from it, and the reader checks them against each other in
both directions. A fact edited in the stored block, and a producer text swapped
under blocks that no longer match it, both refuse, printing both sides.

<a id="fixed-facts-schema-missing-record-refused"></a>
`fixed-facts-schema.missing-record-refused` deletes each half in turn (either
block, or the producer text) and requires refusal with the migration
instruction named. A file that cannot say which cosmology it belongs to is
refused, not served.

<a id="fixed-facts-schema-legacy-version-refused"></a>
`fixed-facts-schema.legacy-version-refused` requires legacy schema versions 1
and 2 to refuse with the migration instruction, a block grammar from the future
to refuse, and the supported version to be accepted — a check that only ever
refuses proves nothing about the file it must let through. The reader accepts
exactly the version declared by `emulator/fixed_facts.py`.

<a id="fixed-facts-schema-sampled-and-fixed-refused"></a>
`fixed-facts-schema.sampled-and-fixed-refused` requires a coordinate both
sampled and held fixed to be refused when the sidecar is composed, naming the
coordinate and both values. Allowed, the two halves of the record would answer
"what was w?" differently depending on which half was read.

<a id="fixed-facts-schema-parameter-order-enforced"></a>
`fixed-facts-schema.parameter-order-enforced` requires a whitening geometry
whose parameter order is a permutation of the record's to be refused, printing
both orders. Counting names, or comparing them as a set, would let a
permutation through — and a permutation silently pairs every incoming value
with the wrong parameter's column, so predictions are confidently wrong and
nothing about the numbers looks unusual.

<a id="fixed-facts-schema-mutation-arms-red"></a>
`fixed-facts-schema.mutation-arms-red` breaks the record's own laws on purpose
and requires the guarding legs to go red: accepting a legacy schema must fail
the version leg, and a stored block edited away from the producer text beside
it must fail the verbatim-copy leg. A valid control confirms the faithful file
still reads. (**Rejected: a chain-level byte digest.** Which rows trained is
recorded by the staged-selection records and which universe an artifact belongs
to by its facts, so a chain digest would only duplicate both.)

<a id="fixed-facts-schema-vertical-law-enforced"></a>
`fixed-facts-schema.vertical-law-enforced` is the basic fixed-value check: when
the artifact and Cobaya's constant-parameter mapping expose a concrete value
under the same name, those values must agree, and the error names both plus the
corrective action. Missing, renamed, derived, and `n/a` values stay unchecked —
Cobaya permits arbitrary reparameterizations, so a name comparison cannot prove
two cosmologies equivalent, and a custom parameterization stays the user's
responsibility.

<a id="fixed-facts-schema-horizontal-law-enforced"></a>
`fixed-facts-schema.horizontal-law-enforced` verifies that artifacts combined
in one prediction record the same fixed cosmology, conventions, and
sampled-coordinate set, refusing a mismatch with the disagreeing fact and both
values named. Two artifacts trained on different draws of one design pass: they
approximate the same physical maps, so serving them together is sound.

<a id="fixed-facts-schema-domain-law-enforced"></a>
`fixed-facts-schema.domain-law-enforced` verifies each requested point before
inference. Both support endpoints are accepted; a point outside is refused with
the interval, requested value, and corrective action; a record with undeclared
support is refused before numeric conversion, naming the synthetic generator,
rather than failing incidentally inside `float("n/a")`.

<a id="fixed-facts-schema-served-support-is-the-intersection"></a>
`fixed-facts-schema.served-support-is-the-intersection` requires a pair's served
support to be the coordinate-wise intersection of both supports, refusing a
point supported by only one artifact and a pair with a disjoint coordinate.
Support therefore stays separate from the fixed-fact block compared by
equality.

<a id="fixed-facts-schema-comparison-laws-are-load-bearing"></a>
`fixed-facts-schema.comparison-laws-are-load-bearing` requires targeted negative
controls: the checks must fail when the fixed-facts comparison is removed, when
undeclared support is accepted, when an outside point is accepted, and when
support union replaces intersection. Direct fixed-value match, mismatch,
missing-name, renamed-name, and `n/a` controls state the deliberately limited
runtime behavior, and the unmodified implementation must pass every valid
control.

<a id="fixed-facts-schema-resolved-model-read-once"></a>
`fixed-facts-schema.resolved-model-read-once` requires dataset generation to
call `fixed_facts.resolved_constants(model)`. Precedence: theory-component
`extra_args` supply initial values, the parameter block overrides duplicate
names, and the first theory component supplies names duplicated across
components. It preserves the concrete names Cobaya exposes and invents no
aliases; booleans stay booleans, numerics become floats; if the model cannot be
inspected, unreadable values stay absent and the record stores `n/a`. Needs
only NumPy and a small model-shaped object.

<a id="cs-adapter-identity-adapter-contract"></a>
`cs-adapter-identity.adapter-contract` proves the cosmic-shear adapter reads its
configuration from the artifacts: the parameters it requires of the chain are
the emulator's own stored geometry names, and the vector it serves is the
section the stored geometry declares (`dv_return: 3x2pt` scatters into the full
layout with zeros off the mask). A wrong-kind artifact — a scalar emulator,
which returns a `{name: value}` dict rather than a vector — is refused by name,
pointing at the adapter it belongs in. This CPU-capable identity leg is
required because the separate `cobaya-adapter` integration gate needs CosmoLike
and a GPU; neither gate may claim the other's capability boundary or evidence.

<a id="cs-adapter-identity-record-laws-refuse"></a>
`cs-adapter-identity.record-laws-refuse` requires the cosmic-shear adapter to
enforce all three comparison laws at their owning boundaries: after
configuration validation, initialization refuses artifacts describing different
universes; when Cobaya supplies the provider, the adapter compares directly
named artifact constants with directly named model constants, treating an
unavailable or renamed value as inconclusive rather than a refusal; and before
encoding each point, `predict` refuses values outside stored support and
records with undeclared support. Each assertion checks law-specific error text,
so an unrelated `ValueError` — including one from `float("n/a")` — cannot
satisfy it.

## Acceptance evidence: geometry module paths

The **gate registry** is the catalog in `ai/gates/board.py`, executed by
`ai/gates/run_board.py`; its `repo_py_files()` returns the repository
Python-file set used by whole-repository checks.

<a id="geo-paths-evidence"></a>
**geo-paths — fresh artifacts name geometry classes from the geometry package,
and the retired flat module paths remain absent.**

- registration: `ai/gates/board.py` owns the manifest, the
  `ai/gates/checks/geo_paths.py` child, and the three leg names. The census leg
  scans every repository Python file from `repo_py_files()` except its own
  check source, which holds the retired names as test data.
- metric: each named leg checks the expected geometry-class module prefix,
  exact attribute count, and finite prediction; the others run a complete
  six-name disk/import census or a repository-Python reference census with the
  one named self-exclusion.
- capability boundary: CPU PyTorch. The child asserts every leg and its exit
  status is the aggregate, not an extra leg. NumPy and HDF5 are ordinary child
  imports;
  if either is absent the child fails before these legs rather than reporting a
  capability skip. A pass of the complete registered acceptance suite is
  separate integration evidence, not a `geo-paths` leg.

<a id="geo-paths-fresh-save-uses-folder-paths"></a>
`geo-paths.fresh-save-uses-folder-paths` requires a fresh artifact to hold at
least two attribute values beginning `emulator.geometries.`, none beginning
with the retired flat prefix, and a finite prediction after rebuild. It
identifies the package prefix but not which two geometry classes own the
markers; no class-specific claim may be derived from it.

<a id="geo-paths-legacy-flat-paths-absent"></a>
`geo-paths.legacy-flat-paths-absent` checks each of the six retired module
names on disk and through `importlib.util.find_spec`; every one must be absent.

<a id="geo-paths-legacy-reference-census"></a>
`geo-paths.legacy-reference-census` scans the complete repository Python-file
set from `repo_py_files()`, excluding only the check that contains the search
terms, and requires zero retired flat-module references.

## Acceptance evidence: wrapper-family gates

The naming and evidence specification for six artifact-lifecycle gates: two
identity children that run in any CPU environment with PyTorch, two smoke
wrappers that read a real training driver's output, and the paired
save/rebuild and Cobaya integration checks, which need CosmoLike and a GPU.

Three declared anchors require executable actions beyond the output checks
their wrappers describe. A log message, or an instruction to inspect a file, is
not acceptance evidence. Unless the named action executes, the wrapper records
the anchor as unavailable or non-passing and never infers a pass from adjacent
output.

<a id="finetune-identity-evidence"></a>
**finetune-identity — a warm-started emulator computes the source emulator's own
function before the first training step, whatever new parameters were added.**

- registration: `ai/gates/board.py` owns the manifest, the
  `ai/gates/checks/finetune_identity.py` child, and the seven leg names. The
  fixture needs no CosmoLike and no scientific dataset.
- metric: exact tensor equality for the encoding, the transferred weights, and
  the degenerate state dict; the parity leg reads the warm-start result line
  (`max|dv| = 0.000e+00` on 256 rows); every refusal case requires the declared
  exception and message.
- capability boundary: none; the child passes in CPU-only PyTorch. Each leg
  is asserted in the child with one `##AID`, and the child's exit status is the
  aggregate result rather than an extra leg.

<a id="finetune-identity-extended-parameter-encoding"></a>
`finetune-identity.extended-parameter-encoding` requires extras `[w0, wa]` in
covariance order, shared coordinates encoding bit-identically to the source,
and extra coordinates unmoved by a shared-only shift.

<a id="finetune-identity-weight-transfer-and-padding"></a>
`finetune-identity.weight-transfer-and-padding` requires the padded keys to be
exactly the input-consuming tensors, every unchanged tensor copied exactly, and
each padded tensor to be the source columns followed by exact zeros.

<a id="finetune-identity-pre-train-parity"></a>
`finetune-identity.pre-train-parity` requires `build_warm_start` to pass and
return its result line, and `load_state_dict(init_state, strict=True)` to
accept the returned state dictionary with no missing or unexpected key.

<a id="finetune-identity-output-geometry-pin"></a>
`finetune-identity.output-geometry-pin` requires a matching
dataset/probe/width to reuse the source geometry object, and a data-vector
width mismatch to raise.

<a id="finetune-identity-degenerate-no-extras-identity"></a>
`finetune-identity.degenerate-no-extras-identity` requires the no-extras case
to leave geometry tensors and transferred state dict exactly equal to the
source: the degenerate warm start is a copy.

<a id="finetune-identity-loud-config-errors"></a>
`finetune-identity.loud-config-errors` requires three raises: a non-superset
parameter set naming the missing source parameter, a `model:` block beside
`finetune:`, and a `--rescale` other than `none`.

<a id="finetune-identity-anchor-mask-and-freedom"></a>
`finetune-identity.anchor-mask-and-freedom` requires the anchor mask to zero
exactly the padded extra columns, the source columns pinned to `init_state`,
the padded extra columns still free, and configured `anchor: 0.0` an exact
no-op.

<a id="transfer-identity-evidence"></a>
**transfer-identity — a frozen base under a zero-output correction predicts the
frozen base itself, in every form and space, and a saved composition reloads to
the same prediction.**

- registration: `ai/gates/board.py` owns the manifest, the
  `ai/gates/checks/transfer_identity.py` child, and the eight leg names. The
  fixture builds plain, factored, and grid bases plus a composed transfer
  artifact, with no CosmoLike and no dataset.
- metric: exact tensor equality for the epoch-0 identity, the base-encoding
  slice, and the save/rebuild composition; `1e-6` for the `EmulatorPredictor`
  comparison; call counting for the base cache; a raise per refusal.
- legs: 8 — `transfer-identity.plain-base-slice-and-identity`,
  `.factored-base-slice-and-identity`, `.zero-init-surgery`,
  `.loud-config-errors`, `.artifact-lifecycle-round-trip`,
  `.refined-base-lifecycle`, `.diagonal-family-composition`,
  `.cross-family-base-refusal` — one `##AID` each. The two legs inside
  `check_diagonal` are emitted by that function rather than around it, so the
  cross-family refusal reports under its own name.
- capability boundary: none. The cross-family fixture uses a plain grid base so
  the family refusal, not the no-chaining refusal, is the only invalid
  condition.

<a id="transfer-identity-plain-base-slice-and-identity"></a>
`transfer-identity.plain-base-slice-and-identity` requires, for a plain base:
extras `[w0, wa]`, the base encoding an exact column slice of the run's
encoding, and every combination in the two-by-two product of correction form
and representation space preserving the target width, the base cache (one base
encode, no chi-square recomputation), and the epoch-zero identity with
independence from the added coordinates.

<a id="transfer-identity-factored-base-slice-and-identity"></a>
`transfer-identity.factored-base-slice-and-identity` requires the same set for
a factored (three-template) base.

<a id="transfer-identity-zero-init-surgery"></a>
`transfer-identity.zero-init-surgery` requires the correction's final `Linear`
to be exactly zero in weight and bias, every other tensor untouched.

<a id="transfer-identity-loud-config-errors"></a>
`transfer-identity.loud-config-errors` requires seven raises: unknown
`transfer.form`; transfer with `--rescale`; with pce; with finetune; with
`model.ia`; an incomplete `refine` block; and a non-superset parameter set.

<a id="transfer-identity-artifact-lifecycle-round-trip"></a>
`transfer-identity.artifact-lifecycle-round-trip` requires a rebuilt transfer
artifact to return the embedded base with its form and space, its composed
prediction to equal the in-memory composition exactly,
`EmulatorPredictor.predict` to agree to `1e-6`, and chaining — a transfer used
as a base — to be refused.

<a id="transfer-identity-refined-base-lifecycle"></a>
`transfer-identity.refined-base-lifecycle` requires a refined artifact's
composed prediction to use the drifted base exactly, and a drifted state
without its companion attribute to raise.

<a id="transfer-identity-diagonal-family-composition"></a>
`transfer-identity.diagonal-family-composition` requires, on the diagonal
families: the epoch-0 identity through the log law for both forms; the packed
target with an exact zero-correction chi2; refusal of physical space; the
transfer-validator resolutions and rejections; the family validators'
acceptance matrix; and a saved grid transfer artifact predicting the
composition exactly.

<a id="transfer-identity-cross-family-base-refusal"></a>
`transfer-identity.cross-family-base-refusal` requires `from_config` to raise a
`ValueError` naming the never-across-families rule when a grid2d run points at
a plain grid base. The fixture violates only this rule, and the leg keeps its
own evidence identifier so a failure names the family refusal rather than the
larger composition group.

<a id="save-rebuild-drift-evidence"></a>
**save-rebuild-drift — an emulator rebuilt from its saved artifact pair reproduces
the live model exactly, its checkpoint is CPU-normalized, and a file the
schema cannot honour is refused.**

- shared output: the child persists its plain variant to
  `<driver_root>/chains/gates_emul_evaluate` for the cobaya-adapter evaluate
  leg, where `<driver_root>` is the configured training-project directory.
- registration: `ai/gates/board.py` owns the manifest, the
  `ai/gates/checks/gsv_bitwise_drift.py` child, and the nine leg names.
- metric: exact tensor equality between live and rebuilt outputs; exact CPU
  device type for every value in the checkpoint's nonempty tensor-only state
  dict; a named-message raise per refusal.
- capability boundary: CosmoLike, a CUDA GPU, and the configured deployment
  data, so the gate is capability-skipped on a CPU-only machine. Only a
  CUDA-trained save can prove CPU normalization moved a tensor rather than
  observing one already there.

<a id="save-rebuild-drift-plain-rebuild-matches-live"></a>
`save-rebuild-drift.plain-rebuild-matches-live` requires the plain variant's
rebuilt output to equal the live model's output exactly for the first eight
validation-input rows. In `gsv_bitwise_drift.py`, `exp` is the experiment
object, `val_set` its validation-data mapping, and `"C"` selects the matrix of
cosmological parameter inputs.

<a id="save-rebuild-drift-cpu-normalized-state"></a>
`save-rebuild-drift.cpu-normalized-state` loads the just-written plain
checkpoint without `map_location` and requires a nonempty dictionary whose
values are all tensors reporting `device.type == "cpu"`. Without load-time
relocation the observed device is the one `save_emulator` serialized, not one
the check selected.

<a id="save-rebuild-drift-factored-rebuild-matches-live"></a>
`save-rebuild-drift.factored-rebuild-matches-live` requires the same for an
`nla` factored save.

<a id="save-rebuild-drift-npce-rebuild-matches-live"></a>
`save-rebuild-drift.npce-rebuild-matches-live` requires the same for a
neural-PCE save.

<a id="save-rebuild-drift-head-rebuild-matches-live"></a>
`save-rebuild-drift.head-rebuild-matches-live` requires the same for a
convolutional-head save, and requires rebuild to reconstruct the residual
convolutional network (`ResCNN`) from the persisted bin split alone, without
reading a dataset configuration file.

<a id="save-rebuild-drift-code-default-drift-ignored"></a>
`save-rebuild-drift.code-default-drift-ignored` rebuilds the plain save in a
child process importing the emulator package from a copied source tree whose
only change is `make_activation`'s `n_gates` default (3 -> 7), written to disk
before the child starts, and requires the output to be unchanged: rebuild reads
the file, not the activation code's source default. The child first proves the
changed default is live and refuses with its own exit code otherwise, so a
launch that imported the ordinary package cannot pass as proof.
`ai/tests/test_drift_gate_child_isolation.py` runs the same helpers on a small
synthetic gated-power artifact, verifying the copy substitution, the bitwise
child comparison, and the unmodified-copy refusal without the workstation data.
This arm claims nothing about compile-mode persistence, because rebuild is
deliberately called with `compile_model=False`.

<a id="save-rebuild-drift-v1-schema-refusal"></a>
`save-rebuild-drift.v1-schema-refusal` forges `schema_version` to 1 and
requires the rebuild to raise with the migration instruction named.

<a id="save-rebuild-drift-v2-schema-refusal"></a>
`save-rebuild-drift.v2-schema-refusal` forges `schema_version` to 2 and
requires the same. A v2 file carried no record of the cosmology it was trained
under, so it cannot prove it belongs to the cosmology it is about to be asked
about; the reader refuses rather than guessing. `emulator/fixed_facts.py` is
the schema-version authority.

<a id="save-rebuild-drift-old-head-artifact-refusal"></a>
`save-rebuild-drift.old-head-artifact-refusal` deletes the persisted bin split
from a head save (a pre-persistence artifact) and requires the rebuild to raise
a `KeyError` naming the bin-split persistence — never to re-derive the split.

<a id="compile-recipe-evidence"></a>
**compile-recipe — a CUDA rebuild consumes the compile mode persisted in its
artifact.**

- fixture design: the `case-a` and `case-b` schema-3 scalar artifact pairs
  are opaque — neither the paths nor nearby labels encode a mode — so the leg
  cannot pass by reading its own fixture names.
- registration: `ai/gates/board.py` owns the manifest, the
  `ai/gates/checks/compile_recipe.py` child, and the two leg names, ordered
  `compile-recipe.observation-controls` then `.cuda-persisted-modes`.
- metric: the independently read saved modes; each ordered `torch.compile`
  call's mode, uncompiled input callable, and returned compiled callable;
  explicit records of an identity compiler result or a rebuild that discards
  the compiler result; compiler or rebuilt-forward exceptions; finite callable
  outputs.
- what each leg proves: the CPU leg writes and reads both real artifacts,
  exercises every rejected observation below, and forges a missing field
  through production rebuild; the CUDA leg records and delegates the real
  compiler call for both artifacts, then binds each compiled result to
  production's exact returned callable.
- capability boundary: the CPU leg needs no CUDA. The CUDA leg needs a
  CUDA-capable environment supporting both modes and executing both
  `compile_model=True` rebuilds. The standalone fixture needs neither CosmoLike
  nor deployment data.

<a id="compile-recipe-observation-controls"></a>
`compile-recipe.observation-controls` requires two real supported-schema saves
to persist the two distinct modes. Its ordered result accepts one matching call
per artifact and rejects a lost call, a duplicate call, an exception from the
compiler or compiled forward pass, a swapped substitution, either hard-coded
mode, an identity compiler result, and a rebuild that discards the compiled
result. Deleting `compile_mode` from an otherwise-valid recipe must make
production `rebuild_emulator` raise a `KeyError` naming the field and the
no-code-default rule.

<a id="compile-recipe-cuda-persisted-modes"></a>
`compile-recipe.cuda-persisted-modes` first runs a tiny CUDA compiled forward
in both modes as the capability boundary, then independently reads each saved
mode, rebuilds each artifact with `compile_model=True`, and records exactly one
matching call through a wrapper delegating to the captured real
`torch.compile`. The delegated result must not be the eager input, production's
rebuild return must be that exact result, and it must produce a finite forward.
After both capability probes succeed, any save, read, rebuild, compile, or
forward exception is a failure rather than a capability limitation. The leg
proves the saved value reaches the real call and production uses its returned
callable; it claims no particular internal PyTorch optimization strategy.

<a id="cobaya-adapter-evidence"></a>
**cobaya-adapter — the predictor a Cobaya theory block calls at sampling time
reproduces the training-side data vector and scatters it into the layout the
likelihood expects.**

- generated dependency: the evaluate leg consumes the emulator
  `save-rebuild-drift` persisted at
  `<driver_root>/chains/gates_emul_evaluate.h5`; the registry owns that through
  `deps=("save-rebuild-drift",)` and runs the prerequisite first.
- registration: `ai/gates/board.py` owns the manifest, the
  `ai/gates/checks/gct_parity.py` child for the parity legs, the `cobaya-run`
  evaluate leg, and all seven leg names.
- metric: worst relative error <= `1e-6` for parity (denominator
  `|want| + 1e-8`); set and length equality for the scattered-vector legs;
  process exit code for the evaluate run.
- leg boundary: the four child parity legs emit one `##AID` each and the
  wrapper executes the two evaluate legs;
  `mcmc-smoke` requires the separate short chain below, which evaluate output
  cannot satisfy.
- capability boundary: CosmoLike, Cobaya, and a GPU. The wrapper reports
  `mcmc-smoke` as unavailable or non-passing unless it executes the short
  chain.

<a id="cobaya-adapter-plain-predictor-parity"></a>
`cobaya-adapter.plain-predictor-parity` requires the `EmulatorPredictor` built
from the saved plain file to match the training-side kept-entry data vector to
a worst relative error of `1e-6` across the first eight validation-input rows.
In `gct_parity.py`, `exp` is the experiment object, `val_set` its
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
`cobaya-adapter.evaluate-emulator-present` requires the emulator produced by
`save-rebuild-drift` to exist on disk before the evaluate run starts. The
registry runs that prerequisite automatically through
`deps=("save-rebuild-drift",)`, including when the user requests only
`--gate cobaya-adapter`. If the file is still absent afterwards, the gate fails
before drawing any conclusion about Cobaya.

<a id="cobaya-adapter-example-evaluate-run-completes"></a>
`cobaya-adapter.example-evaluate-run-completes` requires `cobaya-run` on the
registry's evaluate YAML (the `lsst_y1` likelihood, `use_emulator 1`) to exit
zero. It proves the run completes; numerical parity is established separately
by the named predictor assertion.

<a id="cobaya-adapter-mcmc-smoke"></a>
`cobaya-adapter.mcmc-smoke` requires a real short-chain run that drives the
theory block through a sampler, not merely through `evaluate`. A wrapper that
does not start the sampler records this anchor as unavailable or non-passing;
a successful evaluate run is not substitute evidence.

<a id="finetune-smoke-evidence"></a>
**finetune-smoke — a real fine-tune run continues the gate registry's own saved
emulator.**

- generated dependency: the emulator produced by `save-rebuild-drift`,
  supplied through `deps=("save-rebuild-drift",)`.
- registration: `ai/gates/board.py` owns the manifest, the cosmic-shear
  training driver run on the `finetune-smoke-config` YAML, and the four leg
  names.
- metric: process exit code, plus exact presence of the parity result line and
  the warm-start banner.
- leg boundary: the wrapper obtains three legs from the driver's exit code and
  output; `artifact-provenance-and-round-trip` additionally requires the file
  action below, which output text cannot satisfy.
- capability boundary: CosmoLike and a GPU.

<a id="finetune-smoke-run-completes"></a>
`finetune-smoke.run-completes` requires the fine-tune driver to exit zero.

<a id="finetune-smoke-parity-verdict-printed"></a>
`finetune-smoke.parity-verdict-printed` requires the driver's output to carry
the pre-train parity line (`finetune parity: max|dv|`). This text-presence leg
proves the driver ran the parity check and printed its result; the identity
itself is asserted numerically by finetune-identity.

<a id="finetune-smoke-warm-start-banner"></a>
`finetune-smoke.warm-start-banner` requires the startup banner to announce the
source artifact (`finetune: from `).

<a id="finetune-smoke-artifact-provenance-and-round-trip"></a>
`finetune-smoke.artifact-provenance-and-round-trip` requires the wrapper to
open the saved `.h5`, verify the `finetuned_from` root attribute, rebuild with
`rebuild_emulator`, and compare its prediction with the saved run's reference
prediction. A wrapper that inspects only standard output records this anchor as
unavailable or non-passing. `finetune-identity` tests the general mechanism but
does not replace this file-specific evidence.

<a id="transfer-smoke-evidence"></a>
**transfer-smoke — a real transfer run composes a correction over the gate registry's own
saved base.**

- generated dependency: the plain base produced by `save-rebuild-drift`,
  supplied through `deps=("save-rebuild-drift",)`.
- registration: `ai/gates/board.py` owns the manifest, the cosmic-shear
  training driver run on the `transfer-smoke-config` YAML, and the five leg
  names.
- metric: process exit code, plus exact presence of four driver-output lines:
  the epoch-zero parity result, the transfer banner, and the two save lines.
- leg boundary: the wrapper obtains four legs from the driver's exit code and
  output; `artifact-provenance-and-round-trip` additionally requires the file
  action below, which output text cannot satisfy.
- capability boundary: CosmoLike and a GPU.

<a id="transfer-smoke-run-completes"></a>
`transfer-smoke.run-completes` requires the transfer driver to exit zero.

<a id="transfer-smoke-parity-verdict-printed"></a>
`transfer-smoke.parity-verdict-printed` requires the driver's output to carry
the epoch-zero parity line (`transfer parity: epoch 0 == frozen base`). This
text-presence check proves the driver ran and reported the parity check;
`transfer-identity` separately establishes the numerical identity.

<a id="transfer-smoke-transfer-banner"></a>
`transfer-smoke.transfer-banner` requires the startup banner to announce the
base and its form and space (`transfer: from `).

<a id="transfer-smoke-saved-artifact-paths-printed"></a>
`transfer-smoke.saved-artifact-paths-printed` requires both save lines (`saved
emulator ->` and `saved run record ->`) in the output. It proves the save ran
and printed its two paths; reload behavior belongs to the next leg.

<a id="transfer-smoke-artifact-provenance-and-round-trip"></a>
`transfer-smoke.artifact-provenance-and-round-trip` requires the wrapper to
open the saved `.h5`, verify the `transfer_from` root attribute and embedded
`transfer_base` group, rebuild the artifact, and require the composed
prediction to reproduce the in-memory composition. A wrapper that inspects only
standard output records this anchor as unavailable or non-passing.
`transfer-identity` tests the general lifecycle mechanism but does not replace
this file-specific evidence.
