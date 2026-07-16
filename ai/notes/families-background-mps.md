# Background and matter-power families

This note is the durable engineering contract for background-grid and
matter-power-grid emulators. It records current scientific rules, why they
exist, which module owns each rule, and the evidence required before a change
is accepted. It does not record development sessions, ticket identifiers,
dates, or role verdicts.

## Vocabulary used throughout this note

A polynomial chaos expansion (PCE) is a polynomial surrogate. Neural PCE
(NPCE) combines a frozen PCE base with a neural model that learns its residual.
TRF is the repository abbreviation for a transformer correction head.

The **matter-power spectrum (MPS)** describes how matter fluctuations depend
on physical scale. An **artifact** is a saved emulator publication containing
trained weights and the scientific facts needed to rebuild them. A **schema**
is the required fields, types, and meanings of a saved record.

A **loss ladder** is the supported set of chi-squared, square-root, and reverse
Huber (BerHu) training transforms. For a per-sample chi-squared value, this
BerHu transform follows its square root below a configured knot and becomes
linear in chi-squared above the knot. Its capped form returns to a square-root-
shaped tail above a second knot. **Annealing** changes a configured value
gradually over training epochs. **Trimming** removes a configured fraction of the
largest-error rows before averaging; **focus** gives harder retained rows more
weight. **Clipping** limits a gradient norm before an update, while **rewind**
restores one saved model-and-optimizer snapshot. An **exponential moving
average** is a smoothed copy of model weights that emphasizes recent updates.

A **correction head** is the trainable network part that predicts a residual
or ratio. **Fine-tuning** continues training from saved weights.
**Frozen-base transfer** evaluates a saved base model without changing its
weights and trains only its correction. **Law space** is the output after
applying the family's declared transform, such as a logarithm; the decoder
returns from law space to physical values.

A **bin** is one physically meaningful group of adjacent output coordinates
processed together. A transformer **token** is the vector that represents one
bin or redshift slice inside the model. For a two-dimensional grid,
`data.grid2d.k_stride` retains every `k_stride`-th wavenumber coordinate plus
the upper edge before geometry and head layout. The **post-stride token width**
is the number of retained wavenumber coordinates in each redshift-slice token.
An **attention head** is one independent set of learned query, key, and value
projections; the token width must split evenly among the requested heads.

A **gate** is a registered acceptance command. A **fixture** is its fixed
input setup. A **smoke check** is a short end-to-end run through real
generation, training, saving, and serving. A **tripwire** is a numerical or
structural condition chosen to fail when a named broken behavior returns. The
**central processing unit (CPU)** is the ordinary host processor.

BAOSN names background expansion outputs used with baryon acoustic oscillation
and supernova data. CAMB (Code for Anisotropies in the Microwave Background)
is the external cosmology solver used in real-provider comparisons. Cobaya is
the Bayesian inference program that requests predictions; its `Theory` class
is the interface implemented by a prediction component. A PDF is a Portable
Document Format diagnostic file.

FITPACK is the numerical spline library used through SciPy. Hierarchical Data
Format version 5 (HDF5) is the saved-array file format. Lambda cold dark matter
(LCDM) is the cosmological model with a cosmological constant and cold dark
matter. Random-access memory (RAM) holds arrays while a process is running.
`rtol` and `atol` are the relative and absolute tolerances used by numerical
comparisons.

## Shared family structure

### Rule

Background and matter-power families use the full shared training surface:
loss ladders, annealing, trim/focus, exponential moving average,
clip/rewind, fine-tuning, correction heads, frozen-base transfer, and residual
NPCE where the family schema permits it.

For a one-dimensional grid, one physical bin covers the redshift axis unless a
TRF token count explicitly resegments that single bin. For grid2d, one bin per
redshift slice owns the flattened wavenumber columns. Convolution channels and
TRF tokens therefore follow redshift slices. A requested token count is
rejected when physical bins already define the tokens, and the attention head
count must divide the post-stride token width.

NPCE is residual-only. The polynomial base fits law-space rows; the network
learns the residual. Constant pins apply after base and network composition.

### Why

The family geometry already carries physical coordinate order. Rebuilding an
identity basis wastes memory and creates a second coordinate convention.
Residual NPCE preserves one decode rule and prevents a polynomial base from
competing with the target-law transform.

### Owner

- shared model and phase behavior: model and training modules;
- grid geometry: `emulator/geometries/grid.py`;
- grid2d geometry: `emulator/geometries/grid2d.py`;
- family staging: `emulator/experiment.py`;
- background serving: `cobaya_theory/emul_baosn.py`;
- MPS serving: `cobaya_theory/emul_mps.py`.

## Background family

### Scientific model

#### Rule

The background family uses two artifacts:

1. A Hubble artifact on the supernova redshift window, whose grid starts at
   exactly `z = 0`.
2. A directly trained transverse-comoving-distance artifact on the
   recombination window.

Inside the supernova window, comoving distance is imposed physics computed
from `c/H(z)`. Angular-diameter and luminosity distances follow the flat-model
relations. The interval between the two trained windows is not interpolated or
bridged. A request in that interval refuses and names both supported windows.

Only the Hubble curve is learned on the low-redshift window. Distances are not
independent networks there. The recombination distance is learned directly
because the gap between windows is intentionally unsupported.

The family is flat-only. Nonzero curvature is refused from the resolved global
model at generation and consumption, whether curvature is sampled or fixed.
Artifacts persist the flat-only scientific fact. The legacy curved-distance
formula is not supported.

#### Why

An analytic extension across an untrained redshift desert manufactures
science. Treating radial distance as transverse distance is correct only in a
flat model. Checking only predictor input names misses a fixed global
curvature parameter.

#### Owner

- generation and grids: `compute_data_vectors/dataset_generator_background.py`;
- distance construction: `emulator/background.py`;
- geometry and metadata: `emulator/geometries/grid.py`;
- public protocol: `cobaya_theory/emul_baosn.py`.

### Cumulative integration

#### Rule

`cumulative_simpson` uses composite Simpson values at even nodes. For an odd
node between an accepted pair of intervals, let `i` be that odd node's array
index, let `y[j]` be the integrand evaluated at grid node `j`, and let `h` be
the constant spacing between neighboring grid coordinates. The one-interval
quadratic formula is

```text
h / 12 * (5*y[i-1] + 8*y[i] - y[i+1])
```

The retired half-of-two-interval expression
`h / 6 * (y[i-1] + 4*y[i] + y[i+1])` is mathematically wrong for the odd
node. Interpolation fits only corrected cumulative values.

#### Why

The retired expression gives a linear-function error of `h^2/2` and a
quadratic one-interval value four times the truth. Fitting a cubic through
wrong odd nodes contaminates arbitrary-redshift queries even when even nodes
are exact.

#### Acceptance evidence

Known-answer checks cover every node for constant, linear, quadratic, and
cubic functions. Constant, linear, and quadratic values are near machine
precision at odd and even nodes; even cubic nodes remain exact. A mutation
restoring the retired odd formula must miss the linear and quadratic answers
by a wide margin.

### Redshift-domain integrity

#### Rule

The supernova grid starts exactly at zero. `comoving_distance_grid` and the
Hubble interpolator require `z_grid[0] == 0` and do not extrapolate below the
stored grid. Background artifacts with a positive first supernova node refuse
at load with regeneration and retraining guidance. The advertised window is
derived from the persisted grid only after that invariant is proved.

Redshift grids are one-dimensional, finite, strictly increasing, and
nonnegative. Hubble values are finite and strictly positive. The cumulative
integrand is finite and positive; the resulting comoving distance is finite,
nondecreasing, and exactly zero at redshift zero. Recombination `D_M` rows are
finite, nonnegative, nondecreasing, and shape-matched to their grid.

#### Why

Allowing a grid that begins above zero makes the integral depend on cubic
extrapolation through an interval where the network was never trained.
Positivity alone cannot detect that untrained-domain error.

#### Acceptance evidence

- Zero-starting analytic controls retain their known distances.
- Grids starting above zero refuse in the integration function and artifact
  load path.
- A mutation restoring extrapolation reproduces a finite but wrong distance.
- Zero, negative, NaN, or Inf Hubble values refuse before division.
- Nonmonotonic, malformed, or mismatched grids refuse with coordinates.
- The real-provider comparison is rerun after any integration change.

### Background metadata

#### Rule

One registry beside the grid target-law registry owns the allowed
quantity/unit pairs:

```text
Hubble -> km/s/Mpc
D_M    -> Mpc
```

Configuration validation, `GridGeometry` construction and rebuild, and the
BAOSN adapter read the same registry. Units must be a native string and must
match the quantity. No `str()` coercion is allowed. Target law is a separate
axis; either quantity may use a supported law when its law-domain contract
holds.

For `log_offset`, offset is a finite native non-Boolean real. The shifted
target, transformed law rows, center, and scale are independently finite. A
nonpositive shifted target reports the grid coordinate. A forged nonfinite
offset, center, or scale refuses at rebuild.

#### Why

A producer that accepts arbitrary units can train an expensive artifact that
the consumer will always reject. NaN and Inf evade comparative guards such as
`scale <= threshold`; finiteness must be checked before comparisons.

#### Acceptance evidence

- Valid Hubble and `D_M` pairs pass config, geometry, save/rebuild, and the real
  adapter.
- Swapped, arbitrary, Boolean, numeric, or null units refuse before staging.
- Infinite, NaN, Boolean, and quoted offsets refuse.
- A finite-offset encode/decode control round-trips.
- Producer/consumer disagreement mutations fail.

### Public background protocol

#### Rule

The adapter loads exactly two artifacts, one Hubble and one `D_M`, identified
by persisted quantity rather than list position. Root order is irrelevant.
Both artifacts belong to one canonical dataset and scientific domain and have
compatible sampled coordinates, domains, and fixed facts. A requirement union
never manufactures compatibility between different coordinate systems.

The product-specific domain helper is shared by `must_provide` and runtime
getters:

- Hubble queries lie wholly in the low-redshift Hubble window;
- distance queries lie in either supported distance window;
- pair-distance requests are exact finite `(N,2)` arrays with
  `z1 <= z2`, and each endpoint is serviceable.

Equal endpoints return exactly zero. One-column, three-column, reversed,
empty, or nonfinite pair arrays refuse before flattening. Startup and runtime
give the same verdict and explanation for the same request.

#### Acceptance evidence

- Valid requests at both windows agree with the corresponding artifact and
  derived-distance law.
- Hubble at recombination redshift refuses at startup and runtime.
- Desert, out-of-window, mixed-validity, and malformed pair requests refuse.
- Exact boundary equality and equal-redshift pairs pass.
- Two artifacts with different dataset identity, fixed facts, or sampled
  coordinates refuse before requirements are published.
- Swapped root order passes.
- A mutation restoring union-only compatibility reproduces a finite stitched
  background from inconsistent cosmologies and must fail.

<a id="bsn-identity-evidence"></a>
## Background identity evidence

`bsn-identity` uses synthetic artifacts to exercise integration, distance
construction, grid geometry, saved prediction, adapter behavior, residual
NPCE, and fine-tuning. It claims CPU behavior only; it does not claim real
CAMB or real Cobaya execution.

<a id="bsn-identity-simpson-polynomial-nodes"></a>
### Simpson polynomial nodes

Every node is compared with analytic constant, linear, quadratic, and cubic
integrals. The retired odd formula must fail linear and quadratic controls.
The required odd-point-count guard remains active.

<a id="bsn-identity-distance-pipeline-consistency"></a>
### Independent distance consistency

The interpolation pipeline is compared with adaptive quadrature applied
directly to analytic flat-model `c/H(z)`. The acceptance band is the production
allowance plus the integrator's reported numerical uncertainty. A separate
same-integrator fine-grid comparison is labeled resolution-only. Scaling every
Simpson weight by `0.99` must escape the shared-function comparison but fail
the independent comparison. A nonfinite pipeline value must fail every
acceptance predicate rather than disappear inside `min` or `max` updates.

<a id="bsn-identity-geometry-and-artifact-round-trip"></a>
### Geometry and artifact round trip

The gate checks log-offset transformation at float32 tolerance, exact grid
state, law and domain refusals, and exact saved prediction and family metadata
under log-offset and no-law configurations.

<a id="bsn-identity-adapter-piecewise-contract"></a>
### Adapter piecewise contract

The gate checks the two-window layout, derived-distance and unit relations,
and typed refusals for desert queries, missing pairs, duplicate quantities,
and out-of-window requests.

<a id="bsn-identity-npce-composition"></a>
### NPCE composition

A nonzero polynomial base is fitted. Residual encode/decode algebra and
base-plus-network prediction remain exact before and after save/rebuild.

<a id="bsn-identity-finetune-parity"></a>
### Fine-tune parity

A compatible grid source passes epoch-zero warm-start parity. Metadata and
quantity mismatches refuse before staging.

<a id="bsn-identity-missing-quantity-refused"></a>
### Missing quantity refusal

The fixture contains two otherwise valid grid artifacts with distinct
quantities but no `D_M`. Shared dataset identity is valid so the missing-
quantity guard, rather than an earlier pair-identity guard, is the refusal
being tested.

<a id="bsn-smoke-evidence"></a>
## Background smoke evidence

`bsn-smoke` generates real background fixtures, trains Hubble and
recombination-distance emulators, compares the provider with CAMB, and writes
diagnostics. Process success is never sufficient without file and value
checks.

<a id="bsn-smoke-generated-background-dumps"></a>
### Generated background dumps

Both generation processes exit zero and write both quantities and both axes.
Every Hubble column has relative spread greater than `1e-5` across generated
cosmologies. This tripwire detects a stale provider cache at the dump.

<a id="bsn-smoke-training-collapse"></a>
### Training collapse

Each model's best validation median falls below half the median score of its
own staged mean predictor.

<a id="bsn-smoke-cobaya-vs-camb"></a>
### Cobaya versus CAMB

The real lifecycle serves Hubble, angular-diameter, and recombination-distance
arrays within maximum relative error `0.02` of CAMB. A desert request refuses.

<a id="bsn-smoke-diagnostics-output"></a>
### Diagnostics output

The check requires two grid diagnostic pages and a nonempty PDF larger than
10,000 bytes. This proves output production, not the interpretation of every
curve.

## Matter-power family

### Scientific model

#### Rule

The network learns a correction to an analytic base:

```text
pklin target = log(P_linear / P_linear_base)
boost target = log(B / B_base)
P_nonlinear  = B * P_linear
```

The generator writes raw and base arrays together. Staging forms the law-space
target once from row-aligned files. The geometry never recomputes a
cosmology-dependent base. The predictor returns a law-space surface; the
adapter owns base reconstruction and nonlinear composition.

`emulator/syren_base.py` is the only parameter-mapping and base-function
owner. Vendored Syren formulas remain repository-controlled. A formula change
is a new base identity and requires retraining.

Grid2d rows flatten redshift outermost. Redshift and wavenumber axes are
persisted. Wavenumber thinning persists the retained grid, including the top
edge. The adapter's low-wavenumber nonlinear blend remains the documented
function of `k`.

#### Why

Computing the base in geometry encode/decode would introduce per-batch device
synchronization and a second implementation of the formula being corrected.
Recomputing at serving would make the artifact depend on an unrecorded
external formula version.

### Syren domain

#### Rule

Every active Syren fit has an explicit calibrated cosmology/redshift domain
and sign requirements for logarithm and square-root arguments. Validation runs
before evaluation. A nonpositive log argument or negative radicand refuses and
names the expression and offending scientific point. Silent `abs` reflection
and epsilon continuation are forbidden unless a primary formula source defines
them under a separately named base variant.

Scalar and vectorized formulas agree throughout the accepted domain. Generator
setup validates the configured prior and grid before parallel launch.
Inference validates each requested point.

#### Acceptance evidence

Valid-domain scalar and vectorized grids agree. Every log boundary and a known
negative-radicand point refuse before returning a spectrum. An `abs`-restoring
mutation must fail. No accepted base value is nonfinite. The shipped production
prior and grid are proved inside the calibrated domain or rejected with the
offending corner.

### Power-spectrum interpolator

#### Rule

Construction requires finite one-dimensional `z` and `k` axes with at least
four unique points each, strictly positive `k`, and a finite surface of exact
shape `(len(z), len(k))`. Duplicate axes refuse. Valid unsorted axes are
sorted with named eager index arrays, and the surface is permuted along the
matching axes without mutating caller arrays.

`logP`, `logsign`, and extrapolation bounds have explicit typed schemas.
Extrapolation bounds are finite and positive, with lower bound below the input
minimum and upper bound above the input maximum. The tail is a straight-line
continuation of `log P` versus `log k`, which is a power law in ordinary
space. One named interior fraction owns the tail-node placement.

Queries reject empty, nonfinite, or nonpositive-wavenumber values before
`min`, `max`, `log`, or spline evaluation. Cartesian `grid=True` and paired
`grid=False` output shapes are documented and tested. SciPy's cubic degree
requirement is explicit.

#### Why

NaN compares false against ordinary bounds and can bypass range checks. A NaN
stored extrapolation boundary can admit arbitrary wavenumbers. Short axes and
duplicate coordinates otherwise fail later inside FITPACK with non-contract
errors.

#### Acceptance evidence

- Sorted and unsorted valid grids agree at nodes and interior queries.
- Caller arrays remain unchanged.
- Three-point axes, wrong surface shape, duplicates, NaN/Inf, and nonpositive
  `k` refuse before SciPy.
- Both tails match an independent analytic log-linear continuation.
- Empty and nonfinite queries raise the public range error.
- The smoke range leg checks low/high `k` and `z`, catches only Cobaya's
  `LoggedError`, and verifies coordinate, requested value, and limit in the
  message. A wrong exception class must fail.

### Nonlinear composition and state publication

#### Rule

Linear power and boost have exact shapes and finite strictly positive values.
Their nonlinear product is checked after multiplication; finite factors that
overflow to Inf refuse. A rejected point leaves no partial power-spectrum
state.

`get_Pk_grid` and `get_Pk_interpolator` follow the installed Cobaya signatures,
including `nonlinear=True` when omitted. Explicit linear and nonlinear calls
preserve their branches exactly. A protocol check compares names and defaults
with the installed base class.

`must_provide` validates supported variable pairs, native Booleans, finite
one-dimensional redshift requests, positive finite `k_max`, exact grid-node or
explicit-resampling policy, and the documented extrapolation limit. Repeated
requirements accumulate the union of redshifts and branches and maximum
`k_max` without mutating the caller mapping. Startup and runtime use one
capability helper and one reason.

#### Acceptance evidence

- A finite overflowing product is rejected with no state keys.
- Omitted nonlinear argument equals explicit `True` and differs from explicit
  `False` on separated sentinel values.
- Signatures match installed Cobaya.
- Unsupported Weyl pairs, out-of-domain redshifts, off-node grid requests, and
  excessive `k_max` refuse at startup with zero predictor calls.
- Repeated requests merge and the caller input remains unchanged.
- Removing `must_provide` or restoring the wrong default must fail.

### Constant-column pins

#### Rule

A grid2d constant pin is legal only when all conditions hold:

1. Quantity is `boost`; a constant linear-power column is a partial-dead-dump
   error.
2. The law-space center matches the law identity within a documented
   float32-derived tolerance: raw boost identity is one, Syren residual
   identity is zero.
3. In every redshift row, pinned coordinates form a contiguous prefix from the
   lowest wavenumber.

The whole-surface constant guard remains active. A stored mask is always
present in current artifacts: an explicitly unpinned geometry stores an all-
false mask. The mask is one-dimensional, exact length `nz*nk`, Boolean or
binary uint8, and validated on rebuild. Missing mask state refuses with a
re-save instruction. No parallel policy-version field is needed because
quantity, law, axes, center, and mask determine legality.

Every diagonal residual reduction applies the validated mask after effective
base/correction composition. Pinned coordinates contribute zero residual and
zero gradient to plain, NPCE-residual, transfer, fine-tune, validation,
best-epoch, and diagnostic metrics. Nonfinite corruption refuses before
masking.

#### Why

Low-wavenumber boost identity can be real physics. A constant linear-power
column or arbitrary constant boost is not. Presence-inferred mask state lets
deleting one HDF5 member silently change served science. A metric that ignores
the mask optimizes coordinates that decode later discards.

#### Acceptance evidence

- Valid low-`k` boost identities pin and round-trip under no-law and Syren law.
- Constant pklin, wrong boost values, or a high-`k` pin refuse with coordinates.
- Forged masks refuse at rebuild.
- All-false and valid pinned masks persist explicitly.
- Deleting the mask refuses before a wrong `1 -> 1.25` served-value witness.
- Two standardized predictions differing only at a pin decode identically and
  have equal zero contribution and zero gradient.
- Perturbing an unpinned coordinate recovers its direct squared residual.
- An all-coordinate-sum mutation reports the discarded residual and fails.

### Fixed scientific facts and pair identity

#### Rule

Generation persists every nonsampled cosmology fact that changes a target or
base, including neutrino mass and convention, dark-energy facts, curvature,
radiation/temperature facts, and base implementation identity. The two MPS
artifacts agree on one canonical dataset, generator, sampled coordinate system,
parameter domains, fixed facts, and axes before requirements are published.
Equal axes alone are not provenance.

The adapter compares resolved global values with artifact facts before any
predictor or base execution. A sampled fact is a normal predictor coordinate,
not also a fixed pin. Legacy artifacts lacking the facts refuse with migration
guidance.

#### Acceptance evidence

- Fixed baseline controls preserve numerics.
- Changing a fixed neutrino mass, dark-energy fact, or curvature refuses before
  prediction.
- A pair with equal axes but different dataset identity or coordinate schemas
  refuses.
- Swapped artifact order passes because quantity labels identify members.
- A mutation checking only predictor names or axes must fail.

### Canonical dark-energy coordinates

#### Rule

One resolver shared by generator and adapter accepts either explicit
`(w0, wa)` or `(w0pwa, w0)` and derives `wa = w0pwa - w0`. The `w`/`w0`
alias equality check runs first. If all three values are present,
`w0pwa == w0 + wa` within the documented representation tolerance. An absent
`wa` never defaults to zero when `w0pwa` is present. The resolved coordinate
law is artifact identity. LCDM absence is legal only as an explicit persisted
fact.

The shipped example must either route a complete nonzero-`wa` point correctly
or refuse with migration guidance. Merely removing a dropped-coordinate marker
without fixing the resolver is forbidden because it converts a startup error
into silent wrong science.

#### Acceptance evidence

- Explicit and transformed coordinates produce one base tuple.
- Consistent all-three values pass; inconsistent values refuse.
- Missing information refuses before prediction.
- A real Cobaya route spies the generator and serving Syren tuples and requires
  equality at nonzero `wa`.
- Restoring “missing wa means zero” reproduces a finite multi-percent spectrum
  error and must fail.

### Sigma-eight

#### Rule

The served product must be the conventional top-hat result at
`R = 8 Mpc/h`. Here `R` is the radius of the spherical top-hat window,
`Mpc` means megaparsec, and the dimensionless Hubble parameter is
`h = H0 / (100 km s^-1 Mpc^-1)`. When the wavenumber axis is stored in
`1/Mpc`, the numerical radius in the window is `8/h` Mpc. A literal numerical
radius of `8` is correct only when wavenumber uses `h/Mpc`; mixing these
conventions is forbidden. `cobaya_theory/emul_mps.py::emul_mps.calculate`
already owns `H0` and must pass `h = H0 / 100` to
`cobaya_theory/emul_mps.py::emul_mps._compute_sigma8`, which owns the unit
conversion and the integral. The current `_compute_sigma8` signature has no
`h` argument and still assigns the literal `R = 8.0` while its `k` axis is in
`1/Mpc`. That implementation does not yet satisfy this rule; acceptance
requires the unit-aware `8/h`-Mpc radius rather than a relabeling of the
current result.

Sigma-eight is available only when the artifact supports the calculation. The
stored redshift axis contains exact `z = 0`; a nearby row is never relabeled
as zero. Axes, surface, finiteness, positivity, and shape are validated before
integration. Integration is float64.

The wavenumber domain passes a documented omitted-tail or convergence
criterion tied to the top-hat integrand. Grid length or guessed endpoints are
not completeness. The generator and manifest persist the resolved facts used
by that proof. Final radicand and result are finite and physically valid.

#### Why

Using `z = 0.009` as zero introduces a finite bias. A short positive grid can
produce a finite sigma-eight value that misses most of the integral. The
literal `R = 8` on a `1/Mpc` axis mixes the Mpc and Mpc/h conventions.

#### Acceptance evidence

- Exact-zero wide-grid control matches an independent float64 result.
- A grid beginning at `0.009` or above refuses rather than relabels or reaches a
  SciPy bounds error.
- Low-`k`, high-`k`, and `1..10` grids fail completeness.
- Extending a passing wide grid changes the result only within the recorded
  tail tolerance.
- Nearest-row and `nk >= 8` mutations fail.
- The real-provider value is compared at a known cosmology.

<a id="mps-identity-evidence"></a>
## Matter-power identity evidence

`mps-identity` exercises synthetic geometry, bounded staging and temporary-
file lifecycle, model variants, adapter assembly, config validation, and fine-
tuning. Synthetic base stubs prove assembly, not the real Syren formula.

<a id="mps-identity-geometry-laws-and-pins"></a>
### Geometry laws and pins

The gate checks float32 law round trips, exact state, width/law guards,
scientifically legal pins, whole-surface refusal, and explicit current mask
state through save/rebuild.

<a id="mps-identity-bounded-staging-values"></a>
### Bounded staging values

Stored rows and their mean are compared with an independent float32-payload
reference. Positivity, bounded reads, disk/RAM choice, whole-selection, and
mean-before-cast mutations are covered. Both `rtol` and `atol` are explicit in
discriminating comparisons.

<a id="mps-identity-stable-streamed-moments"></a>
### Stable streamed moments

Centers and population standard deviations agree with NumPy across uneven
chunks and high offsets. Relative pin boundaries and `from_stats` encoding
agree with materialized law-space encoding.

<a id="mps-identity-staging-file-lifecycle"></a>
### Staging-file lifecycle

Restaging removes the superseded file, a multi-point sweep keeps at most one
live train file, failures remove partial files, validation ownership stays
independent, and resident mode produces no temporary file.

<a id="mps-identity-saved-model-variants"></a>
### Saved model variants

The gate checks exact saved predictions for supported base/law variants,
correction-head phase behavior, residual NPCE algebra, save/rebuild equality,
and refusal of unsupported ratio composition.

<a id="mps-identity-adapter-assembly-and-defaults"></a>
### Adapter assembly and defaults

Synthetic bases prove linear and nonlinear composition, low-wavenumber blend,
grid and interpolator readback, installed default signatures, requirements,
point rejection, and pair/quantity/grid refusals.

<a id="mps-identity-config-and-finetune"></a>
### Config and fine-tune

Current pairing and transfer configs pass. A compatible grid2d source passes
epoch-zero parity; metadata mismatch refuses before staging.

<a id="mps-smoke-evidence"></a>
## Matter-power smoke evidence

`mps-smoke` generates real-CAMB law-none fixtures, trains linear and boost
emulators, compares the provider with CAMB, and writes diagnostics. It does not
claim to execute the real Syren hybrid.

<a id="mps-smoke-generated-power-dumps"></a>
### Generated power dumps

Both generators exit zero and write linear power, boost, and both axes. Every
nonfailed linear-power row is positive and has the expected shape.

<a id="mps-smoke-training-collapse"></a>
### Training collapse

Each model's best validation median falls below half the median score of its
own staged mean predictor.

<a id="mps-smoke-cobaya-vs-camb"></a>
### Cobaya versus CAMB

The real lifecycle serves linear and nonlinear power within maximum relative
error `0.05` of CAMB. Range refusals use the public exception contract.

<a id="mps-smoke-diagnostics-output"></a>
### Diagnostics output

The gate checks grid2d diagnostic shapes, one to three redshift slices, two
pages, and a nonempty PDF larger than 10,000 bytes. This does not certify the
interpretation of every plotted curve.

## Shared adapter mechanics

### Rule

A shared helper module may own device resolution, compile policy, typed option
validation, and repository-root path expansion. Each family remains a separate
explicit Cobaya `Theory` class that owns its own initialize, requirement,
calculate, and getter behavior. Shared helpers do not justify a parameterized
adapter superclass.

### Acceptance evidence

All family identity gates preserve valid behavior. Unknown or mistyped options
refuse more precisely. Device fallback and compile choices agree across
adapters without changing family protocols.

## Claims that must remain narrow

- A synthetic adapter gate does not prove real Syren formulas.
- A law-none smoke gate does not prove the Syren hybrid.
- A background dump-variance tripwire catches whole-dump staleness, not every
  sparse stale row; lifecycle acceptance and payload identity remain required.
- Axis equality is not dataset provenance.
- A manifest authenticates facts; it does not replace scientific-domain
  validation.
- Constant-column numerical detection and scientific permission are separate
  rules.
- Real-CAMB comparisons are rerun whenever served mathematics changes.
