# Scalar and cosmic microwave background (CMB) families

This note is the durable engineering contract for scalar derived-parameter
emulators and CMB-spectrum emulators. It records scientific and software rules,
the reason for each rule, the module that owns it, and the evidence required
for acceptance. It does not record development sessions, ticket identifiers,
dates, role verdicts, or merge history.

## Vocabulary used throughout this note

A polynomial chaos expansion (PCE) is a polynomial surrogate. Neural PCE
(NPCE) combines a frozen PCE base with a neural model that learns its residual.
`H0` is the present-day Hubble expansion rate. A PDF is a Portable Document
Format diagnostic file.

An **artifact** is a saved emulator result containing trained weights and
the facts needed to rebuild them. A **sidecar** is a small companion file that
records names, order, or axes for a larger table. A **schema** is the required
fields, types, and meanings of a saved record. **Provenance** is the saved
record of the exact inputs and source artifact that produced a result. An
**identity** is the saved set of facts or byte fingerprints used to decide
whether two datasets, runs, or artifacts are the same; it is not the
mathematical identity matrix.

A **population scale** is the population standard deviation of one output
column, calculated with the number of rows rather than one fewer in the
denominator. **Standardization** subtracts that column's mean and divides by
its population scale. **Float32 epsilon** is the gap between `1.0` and the
next representable float32 number and measures the format's local resolution.
A **diagonal chi-squared** score sums squared standardized residuals without
cross-coordinate terms. **Whitening** is a more general transformation that
makes a residual's covariance the identity, so an ordinary sum of squares has
the intended covariance-weighted meaning.

A **convolution head** predicts a correction by applying local filters along
an ordered coordinate axis. An **attention head** predicts with learned
comparisons among coordinate tokens. A **fine-tuning** run continues training
from saved weights. Scalar outputs have names but no ordered physical axis, so
the axis-dependent heads are not valid for scalar geometry.

A **gate** is a named validation job whose required result is written before
it starts. A **fixture** is its fixed input setup. A **smoke check** is a
short end-to-end run through real
generation, training, saving, and serving. A **control** is a valid case that
must pass. A **mutation** deliberately restores forbidden behavior and must
fail. **Catch power** is the demonstrated ability of a check to fail for that
mutation. The **central processing unit (CPU)** is the ordinary host
processor.

Cobaya is the Bayesian inference program that requests predictions from these
families. Hierarchical Data Format version 5 (HDF5) is the saved-array file
format. `rtol` and `atol` are the relative and absolute tolerances used by
numerical comparisons. Lambda cold dark matter (LCDM) is the cosmological
model with a cosmological constant and cold dark matter.

The four CMB spectrum names used here are:

- `TT`, the temperature-temperature spectrum;
- `TE`, the temperature and E-mode polarization cross-spectrum;
- `EE`, the E-mode-polarization spectrum;
- `PP`, the gravitational-lensing-potential spectrum.

FIRAS names the Far Infrared Absolute Spectrophotometer whose measured CMB
temperature sets the conversion convention. `muK2` means microkelvin squared,
the stored unit for temperature and E-mode spectra. `FIRASmuK2` requests a
microkelvin-squared conversion using the saved FIRAS temperature. **Raw** CMB
spectra are the stored `C_l` values at multipole `l` before multiplication by
the requested multipole plotting factor.

CAMB (Code for Anisotropies in the Microwave Background) is the external
cosmology solver used for reference spectra. `LMAX` is the largest multipole
included by a check or saved table. An application programming interface (API)
is the documented set of calls through which one module uses another. Positive
semidefinite (PSD) means that no vector gives a negative quadratic form. Full
width at half maximum (FWHM) is the beam-width measure used in covariance
production.

## Ownership map

| Subject | Primary owner |
|---|---|
| Scalar target geometry | `emulator/geometries/scalar.py` |
| CMB target geometry | `emulator/geometries/cmb.py` |
| Scalar metric | `emulator/losses/scalar.py::ScalarChi2` |
| CMB diagonal, factored, and roughness metrics | `emulator/losses/cmb.py` |
| Training/staging configuration | `emulator/experiment.py` |
| Scalar public protocol | `cobaya_theory/emul_scalars.py` |
| CMB public protocol | `cobaya_theory/emul_cmb.py` |
| CMB data generation | `compute_data_vectors/dataset_generator_cmb.py` |
| CMB covariance production | `compute_data_vectors/compute_cmb_covariance.py` |
| Gate claims | `ai/gates/checks/` and the stable anchors in this note |

## Scalar family

### Scientific and geometry contract

#### Rule

One scalar emulator learns named derived parameters from the same parameter
table that supplies its sampled inputs. Scalar training has no data-vector
file. `ScalarGeometry` stores output names, center, and population scale.
Standardization uses the relative zero-variance guard
`8 * float32_epsilon * abs(center)`. A truly constant output refuses. A tiny
but genuinely varying output remains legal when its standardized spread is
well formed.

Input names come from the covariance header. Output names come from
`data.outputs` and must be derived columns in the required `.paramnames`
sidecar. All column selection is by name; no fixed slice or “last column”
assumption is allowed.

Scalar loss is diagonal unit-variance chi-squared in standardized output
space. Scalar outputs have no physical coordinate axis, so convolution and
attention correction heads do not apply. Residual NPCE and ordinary fine-
tuning remain supported because neither requires a coordinate axis.

#### Why

Absolute zero-variance thresholds can miss constants whose apparent spread is
only rounding noise around a nonzero mean. Positional column slicing can train
on a derived decoy or omit a sampled parameter while every table width remains
plausible.

#### Acceptance evidence

- Exact geometry state and named prediction survive save/rebuild.
- A constant output refuses.
- A tiny varying output has standardized spread within `0.05` of one.
- Duplicate sidecar names and wrong derived markers refuse.
- A correction-head architecture refuses for scalar geometry.
- Zero, one, and multiple interleaved derived columns select only the requested
  named outputs.

### Scalar driver

#### Rule

`driver/scalar_train_emulator.py` is the scalar driver. It reads only scalar-legal
keys. The run tag and artifact metadata use parameter tables, ordered input and
output names, and row counts. They do not read or invent `train_dv` or
`val_dv`. A scalar config containing those keys is a schema error.

Artifact provenance names both parameter tables and the ordered scientific
coordinates. A placeholder data-vector field is forbidden.

#### Why

The cosmic-shear driver needs data-vector paths while scalar training does not.
Making the scalar path tolerate an irrelevant key hides a wrong-family config
rather than teaching the family boundary.

#### Acceptance evidence

The ordinary scalar example passes config validation, staging, geometry,
training, save, rebuild, and named prediction. A mutation routing it through a
data-vector run-tag or attribute builder refuses before training.

### Scalar public protocol

#### Rule

`emul_scalars` publishes the union of artifact output names through Cobaya's
derived namespace, `state["derived"]`. It never writes arbitrary output names
as top-level state keys. `state["params"]` and
`state["dependency_params"]` remain untouched. `get_param` reads the same
derived mapping Cobaya exposes.

`provides` in YAML is an optional check-only list. Every listed name must
belong to the artifact union, but the list never hides another artifact
output. An empty list makes no additional assertion. Duplicate outputs,
unknown names, input/output overlap, and wrong-family artifacts refuse.
Multi-artifact calculation is atomic: all predictions are validated in a
local mapping before state is modified.

Output names are native nonempty strings and cannot collide with Cobaya-owned
reserved names. Namespace publication is the primary safety mechanism; the
name check is defense in depth.

Every prediction request names its parameters. A mapping is reordered by the
emulator's stored names. A `(names, values)` pair is checked against those
names. A bare numeric row refuses because length cannot detect a permuted
scientific coordinate system.

#### Acceptance evidence

- Ordinary scalar requests, a check-only subset, and an empty check list pass;
  the adapter still advertises and publishes the complete artifact union.
- Duplicate output, unknown check name, chaining, and wrong-family cases
  refuse.
- Reserved output names refuse during config validation and through a real
  Cobaya lifecycle.
- A failing second predictor leaves no partially published first result.
- Cache and dependency dictionaries remain unchanged.
- Mapping and correctly ordered pair predictions are bitwise equal.
- Bare rows, permuted pairs, and foreign names refuse independently.

<a id="scalar-identity-evidence"></a>
## Scalar identity evidence

`scalar-identity` uses synthetic scalar artifacts to exercise geometry,
save/rebuild, public adapter behavior, residual NPCE, and fine-tuning. It
claims CPU behavior only and does not claim a real Cobaya lifecycle.

<a id="scalar-identity-artifact-round-trip"></a>
### Artifact round trip

Named predictions, scalar geometry tensors, and rebuilt family flags remain
exact before and after save/rebuild.

<a id="scalar-identity-geometry-and-schema-guards"></a>
### Geometry and schema guards

The gate rejects constant outputs, duplicate declarations, and correction
heads while accepting a genuinely varying tiny-magnitude output.

<a id="scalar-identity-scalar-adapter-contract"></a>
### Adapter contract

The gate checks output union and input set, accepts an explicit subset, and
rejects duplicate outputs, chaining, unavailable supersets, and non-scalar
artifacts.

<a id="scalar-identity-npce-composition"></a>
### NPCE composition

A nonzero polynomial base is fitted. Residual encode/decode and
base-plus-network prediction remain exact through save/rebuild.

<a id="scalar-identity-finetune-parity"></a>
### Fine-tune parity

The gate checks epoch-zero warm-start parity, the anchor mask for an appended
input column, and pre-staging refusal of output-order or family mismatch.

<a id="scalar-identity-prediction-names-are-proved"></a>
### Prediction names are proved

The gate requires bitwise equality between mapping and correctly ordered pair
forms. Bare values, permuted names, and foreign names each refuse with a
specific explanation. A length-only mutation must fail.

<a id="scalar-smoke-evidence"></a>
## Scalar smoke evidence

`scalar-smoke` trains an emulator for
`omegamh2 = omegam * (H0/100)^2`, compares with that analytic relation, runs
a real Cobaya evaluate command, and writes diagnostics. Every claim combines
process, row, and value evidence.

<a id="scalar-smoke-fixture-rows-disjoint-and-aligned"></a>
### Fixture rows are disjoint and aligned

The training fixture has 4,000 rows from seed 1234 and validation has 1,000
rows from seed 5678. Every target equals the analytic value of its own stored
float32 row. Neither source repeats a physical row, and train/validation row
sets are disjoint before geometry construction.

<a id="scalar-smoke-same-seed-overlap-refused"></a>
### Same-seed overlap refusal

Regenerating validation with seed 1234 creates 1,000 overlapping rows and must
refuse before training.

<a id="scalar-smoke-window-banner-and-rows-match"></a>
### Window banner and rows match

An independently calculated physical-window mask and seeded selection order
must equal both the reported count and exact staged row identities. The fixture
selects three rows from five eligible rows.

<a id="scalar-smoke-banner-only-mutation-rejected"></a>
### Banner-only mutation

A plausible count paired with three wrong rows must fail the joint count-and-
identity predicate.

<a id="scalar-smoke-training-beats-mean-predictor"></a>
### Training beats the mean predictor

The measured mean-predictor median is `0.489362046123`. The trained validation
median must be below half that value, `0.244681023061`. The reference trained
median is `0.196647360921`.

<a id="scalar-smoke-analytic-prediction"></a>
### Analytic prediction

An off-center prediction is compared with the analytic value. The measured
two-epoch relative-error bar is `0.111893762112`.

<a id="scalar-smoke-dead-network-rejected"></a>
### Dead network rejection

The mean-only predictor has validation median `0.489362046123` and off-center
relative error `0.136626311478`; both exceed their acceptance bars.

<a id="scalar-smoke-diagnostics-output"></a>
### Diagnostics output

The check requires three scalar diagnostic pages and a nonempty PDF larger
than 10,000 bytes. This proves output production, not interpretation of every
curve.

<a id="scalar-smoke-cobaya-evaluate"></a>
### Cobaya evaluate

The Cobaya subprocess exits zero and its derived `omegamh2` readback agrees
with the analytic value within `0.111893762112`. Exit status alone is not
evidence.

## CMB family

### Output geometry

#### Rule

One emulator learns one spectrum among TT, TE, EE, or PP on the exact
consecutive multipole axis `2..lmax`. Multipoles zero and one are not training
coordinates because their zero variance would poison whitening.

`CmbDiagonalGeometry` stores spectrum, exact integer multipoles, center,
per-multipole scale, fiducial spectrum, units, amplitude law, and the semantic
amplitude and optical-depth roles. It does not materialize a dense diagonal
matrix.

Construction and rebuild require a nonempty one-dimensional integer axis equal
to `np.arange(2, ell[-1] + 1)`. Duplicates, gaps, reordering, fractional
values, alternate starts, or mismatched center/scale/fiducial/model widths
refuse. Adapter validation repeats the invariant before an `lmax` request can
be treated as coverage.

Generator checkpoints save the configured consecutive axis beside the four
spectrum stores as `<data-vector-root>_ell.npy`. The sidecar is exact
one-dimensional `int64` data. Resume and append require that file and validate
it before loading TT, TE, EE, or PP arrays. A missing, shifted, reversed,
gapped, repeated, fractional, or same-width wrong axis refuses without
changing any checkpoint file.

#### Why

Maximum multipole alone does not prove coverage. A gapped artifact could
silently serve zeros at missing multipoles, indistinguishable from a physical
prediction.

#### Acceptance evidence

Valid `2..L` controls pass. Gapped, start-at-three, duplicate, descending,
fractional, and width-mismatch cases refuse with the first offending
multipole. Two spectra may have different complete maxima and satisfy
independent valid requests. A same-shaped HDF5 axis mutation must be caught
before serving. On the generator side, fresh allocation writes the exact
sidecar and checkpoint load refuses a missing or mismatched sidecar before
reading spectra (`_load_axis_checkpoint` in
`compute_data_vectors/generator_core.py`, exercised by the `cmb-smoke`
gate).

### Gaussian whitening

#### Rule

The covariance producer calculates a variance for each spectrum coordinate.
For the TT and EE auto-spectra,

```text
V_l^XX = 2 * (C_l^XX + N_l^XX)^2 / ((2*l + 1) * f_sky)
```

where `XX` is TT or EE. The TE and PP variances are

```text
V_l^TE = ((C_l^TT + N_l^TT) * (C_l^EE + N_l^EE)
          + (C_l^TE + N_l^TE)^2) / ((2*l + 1) * f_sky)
V_l^PP = 2 * (C_l^PP)^2 / ((2*l + 1) * f_sky)
```

Here `l` is the integer CMB multipole, `C_l^XY` is the saved fiducial signal
spectrum for fields `X` and `Y`, `N_l^XY` is the corresponding instrumental-
noise power, and `f_sky` is the observed fraction of sky. The inverse variance
and whitening scale are `covinv_l = 1 / V_l` and `sigma_l = sqrt(V_l)`.
The displayed PP equation is the repository's V1 production policy: PP
reconstruction noise is omitted, and the producer records that omission in
the `pp_noise_n0` provenance field. It is not a general PP covariance formula.
Any policy that supplies PP reconstruction noise must add the declared noise
term consistently, change the persisted policy identity, and rebuild the
covariance artifact.

`emulator/geometries/cmb.py::CmbDiagonalGeometry.from_fiducial` is a synthetic
zero-noise, full-sky helper. In that restricted case it uses

```text
sigma_l = C_l_fid * sqrt(2 / (2*l + 1))
```

`C_l_fid` is the helper's fiducial auto-spectrum value at multipole `l`.

Production values come from
`compute_data_vectors/compute_cmb_covariance.py::gaussian_blocks`; the
synthetic helper must not be described as the noisy production formula.

Noise follows the declared beam and map-noise convention. Whitening by
`sigma_l` makes plain sum of squared standardized residuals the Gaussian
chi-squared. The retired variance-as-inverse and reciprocal-factor formulas
are forbidden.

Reference values are finite native non-Boolean reals with domain-specific
positivity. Geometry state validates those values on construction and rebuild.

#### Acceptance evidence

The gate checks exact persisted references, transform round trip, endpoint
scale relation, and typed refusal of nonpositive or nonfinite values. A
nonpositive fiducial control identifies its multipole.

### Amplitude law and metric

#### Rule

The supported factored law is named `as_exp2tau_ref` and uses the dimensionless
order-one factor

```text
f = (A_s_ref / A_s) * exp(2 * (tau - tau_ref))
```

`f` equals one at the persisted fiducial. `as_ref` and `tau_ref` are required
finite config facts; `as_ref > 0`. They persist as float64 numbers in geometry
state and have no code fallback. The retired raw
`exp(2*tau)/A_s` law refuses with retraining guidance. No separate version
field duplicates the law name.

The invertible target transform remains factored, but physical chi-squared
divides the standardized residual by `f` before squaring. Roughness uses the
same factor-corrected physical residual. Every public diagnostic supplies the
same batch's parameters explicitly. A mutable training stash is private to the
immediate loss reduction and is cleared in `finally`.

The amplitude and optical-depth roles are semantic, not merely two distinct
columns. `as_name` and `tau_name` are native nonempty distinct strings,
resolve to distinct indices, and map through the canonical role registry to
raw linear amplitude and optical depth. The factor evaluated through resolved
roles at the recorded fiducial equals one. Swapped or unrelated columns refuse
at config and rebuild. Staging and loss construction use one shared resolved
mapping.

#### Why

Without division, the reported score is `f^2` times physical chi-squared and
changes with amplitude at fixed physical error. The retired factor is roughly
`1e9`, creating an arbitrary target scale. Distinct column names alone do not
prevent swapped scientific roles.

#### Acceptance evidence

- Fiducial factor equals exactly one.
- Primary-amplitude scaling preserves the intended law-space spectrum.
- Fixed physical residuals give invariant chi-squared across amplitude and
  optical depth; the retired form misses by exactly `f^2`.
- Roughness is invariant under the factor.
- Missing batch parameters refuse; stale same-shaped or wrong-length private
  state cannot affect diagnostics.
- Analytic two-row diagnostic scores are `[3,3]`; an omitted-parameter mutation
  reproduces `[12,0.75]` or a shape error.
- Missing reference state, retired law, same-role, same-index, swapped-role,
  and unrelated-role cases refuse.
- Valid registered aliases and nonfiducial known answers pass.

### CMB roughness

#### Rule

Optional roughness is a double-boxcar high-pass on the factor-corrected
whitened residual, never on prediction alone. A boxcar is a moving average
over a fixed number of neighboring multipoles. Applying it twice makes the
triangular smooth component; subtracting that component leaves the high-pass,
short-period remainder. `period_cut` is the configured multipole period below
which oscillations receive the penalty, while `lam` is the positive weight
that adds the roughness score to physical chi-squared. The acoustic scale is
the much wider, roughly 200-to-300-multipole spacing associated with physical
CMB peaks. The cutoff must remain well below that scale. Per-sample total score
is physical chi-squared plus weighted roughness before the one shared
reduction. Phase-specific blocks that cannot support the penalty refuse.

#### Why

Prediction-side smoothness would suppress real acoustic structure and could
mimic a lensing-amplitude signal. A bare second difference has no controlled
band edge.

#### Acceptance evidence

The gate checks frequency discrimination, exact zero for zero residual,
bitwise identity when disabled, one-reduction composition, and a bounded
lensing-period contribution.

### CMB public protocol

#### Rule

The adapter serves Cobaya's documented `get_Cl` contract. Dump temperature and
unit convention are scientific facts. Raw `muK2` output remains exact. For a
fixed `TCMB`, conversion uses the temperature saved with the artifact. When
`TCMB` is a sampled input, conversion uses the exact accepted value for the
current prediction. A missing, Boolean, nonfinite, or nonpositive temperature
refuses before prediction state is published.

For TT, TE, and EE, let `T` be that temperature in kelvin. The stored `muK2`
amplitude factor is `T * 1e6`. The requested factors are `1`, `T * 1e6`, `T`,
`2.7255e6`, and `2.7255` for units `1`, `muK2`, `K2`, `FIRASmuK2`, and
`FIRASK2`. Multiply a stored spectrum by the square of the requested factor
divided by the stored factor. PP remains dimensionless. Default `FIRASmuK2`
is therefore never assumed equal to stored `muK2` by coincidence.

Spectrum-specific `ell_factor` is
`l(l+1)/(2*pi)` for TT, TE, and EE and
`[l(l+1)]^2/(2*pi)` for PP. Multipoles zero and one remain explicit zeros.
`must_provide` and `get_Cl` use one capability helper and give the same verdict
for identical requests. A legacy artifact without a temperature/unit fact
refuses with migration guidance.

Predicted spectra are validated before publication:

- TT, EE, and PP are physically nonnegative at stored multipoles;
- TE remains signed;
- where TT, TE, and EE are all present,
  `TE^2 <= TT * EE` within a representation-derived rounding band;
- values are never clipped or replaced by absolute values;
- a failure leaves no partial `state["Cl"]`.

#### Acceptance evidence

- Raw `C_l` requested in `muK2` remains exact.
- TT/TE/EE and PP factor known answers pass at several multipoles.
- FIRAS conversion matches both fixed-temperature and sampled-temperature
  known answers.
- Real consumer lifecycles using default units or `ell_factor=True` pass when
  supported.
- Missing or forged convention refuses before calculation.
- Negative auto spectra and an impossible `TT=EE=1, TE=2` triplet refuse.
- Signed-TE and positive-semidefinite boundary controls pass.
- Partial-state absence is asserted after failure.

### CMB model variants

#### Rule

CMB correction heads operate in physical multipole order without a synthetic
identity matrix. A single CMB grid is one physical bin; an explicit token count
may divide it into contiguous near-equal windows. Epoch-zero correction is the
identity, and phase controls own trunk/head gradient behavior.

Residual NPCE is supported only when its target construction is compatible
with the amplitude-law contract. Fine-tuning pins spectrum, law and role
columns, exact multipole grid, covariance identity, and fiducial references.
Frozen-base transfer uses the same whitening pins.

#### Acceptance evidence

Correction-head attachment, identity initialization, phase behavior,
save/rebuild equality, token-range errors, NPCE composition, and fine-tune
parity are all executable. Missing persisted coordinate split refuses rather
than deriving it from external files at inference.

<a id="cmb-identity-evidence"></a>
## CMB identity evidence

`cmb-identity` uses synthetic artifacts to exercise diagonal geometry,
amplitude-dependent score, saved prediction and adapter behavior, model
variants, fine-tuning, and a direct non-Gaussian covariance known answer. It
claims CPU behavior only.

<a id="cmb-identity-geometry-and-reference-schema"></a>
### Geometry and reference schema

The gate checks Gaussian scale, persisted fiducial references, geometry state,
the intended endpoint relation, and typed reference refusals. It does not
claim monotonic scale across the whole axis.

<a id="cmb-identity-amplitude-law-and-score"></a>
### Amplitude law and score

The gate checks order-one factor construction, transform round trip, physical
score, factor-corrected roughness, stale-parameter isolation, semantic role
resolution, and mutations restoring the retired raw factor or uncorrected
metric.

<a id="cmb-identity-artifact-and-adapter-round-trip"></a>
### Artifact and adapter round trip

The gate checks exact saved prediction, shared axes, low-multipole padding,
requirements, convention guards, spectrum uniqueness, and range refusals.

<a id="cmb-identity-roughness-contract"></a>
### Roughness contract

The gate checks frequency discrimination, zero residual, disabled identity,
one-reduction composition, and bounded synthetic lensing-period contribution.

<a id="cmb-identity-model-variant-composition"></a>
### Model-variant composition

The gate checks correction-head attachment and phase behavior, save/rebuild
equality, exact residual NPCE algebra and prediction, roughness composition,
and incompatible law/PCE refusal.

<a id="cmb-identity-finetune-parity"></a>
### Fine-tune parity

The gate accepts a compatible CMB source, checks epoch-zero parity, and refuses
a wrong-family or wrong-pin source before staging.

<a id="cmb-identity-covariance-known-answer"></a>
### Covariance known answer

All six non-Gaussian blocks are compared with a direct sensitivity-matrix
contraction at relative error below `1e-9`. The retired weights miss by more
than six orders of magnitude. Raw and scaled lensing-potential conventions are
distinct and have an `L`-dependent ratio. A constant-response wide-band
projection and an exactly zero physical band are covered.

<a id="cmb-smoke-evidence"></a>
## CMB smoke evidence

`cmb-smoke` generates real spectra, builds Gaussian and non-Gaussian
covariances, trains, serves through Cobaya, and writes diagnostics. Process
success is combined with file, schema, value, and lifecycle checks.

<a id="cmb-smoke-generated-spectrum-dumps"></a>
### Generated spectrum dumps

Both generation processes exit zero, all parameter sidecars and four spectra
exist, TT has the expected shape, and PP contains a nonzero value.

<a id="cmb-smoke-gaussian-covariance"></a>
### Gaussian covariance

The covariance subprocess exits zero and writes an exact `2..LMAX` axis with
positive TT standard deviations.

<a id="cmb-smoke-nondiagonal-covariance-structure"></a>
### Non-diagonal covariance structure

Six dense blocks have expected shapes. TT is symmetric, no diagonal falls
below its Gaussian counterpart beyond the stated `1e-10` relative allowance,
at least one off-diagonal is nonzero, the spectrum is nonnegative within its
tolerance, and provenance includes step and fractional-amplitude facts. This
is structural real-CAMB evidence, not the independent numerical known answer.

<a id="cmb-smoke-training-collapse"></a>
### Training collapse

The best validation median falls below half the staged mean-predictor median,
using the same validation batch's parameters in the amplitude-dependent score.

<a id="cmb-smoke-cobaya-serving"></a>
### Cobaya serving

The real lifecycle serves finite padded TT values matching the saved predictor
on `2..LMAX` at `rtol=1e-6`, with exact zeros below two.

<a id="cmb-smoke-diagnostics-output"></a>
### Diagnostics output

The check requires two CMB diagnostic pages and a nonempty PDF larger than
10,000 bytes. This proves output production, not interpretation of every
curve.

## CMB covariance production

### Exact non-Gaussian contraction

#### Rule

The non-Gaussian lensing covariance uses fractional band-amplitude
derivatives. For output spectra `s` and `t`,

```text
N^{s,t}_{l,l'} = sum_b D^s_{l,b} * w_b * D^t_{l',b}
D^s_{l,b} = dC^s_l / dA_b
```

At band width one, band `b` contains one lensing multipole `L` and
`w_b = 2 / ((2L + 1) * f_sky)`. For a wider band under the documented
constant-absolute-response approximation,

```text
w_b = sum_L[2*C_L^2/((2L+1)*f_sky)] / (sum_L C_L)^2
```

Here `l` and `l'` identify two output-spectrum multipoles whose covariance is
being calculated, while `s` and `t` name the two spectra. `L` is a lensing-
potential multipole. A **band** `b` is one declared contiguous group of `L`
values perturbed together, and `A_b` is that band's dimensionless fractional
amplitude. `C_L` is the raw physical lensing-potential spectrum, `D^s_{l,b}`
is the response of output `C^s_l` to `A_b`, and `f_sky` is the observed
fraction of sky. `N^{s,t}_{l,l'}` is the resulting non-Gaussian covariance
between the two output coordinates. For a wider band, `w_b` is the effective
band-amplitude variance and both sums in its formula run over that band's `L`
values. This wider-band projection assumes that the absolute response
`dC_l^s / dC_L^PP` is constant across the `L` values in the band. A constant
fractional response is a different assumption and does not justify this
weight.

A band with zero summed physical power contributes zero. The derivative
coordinate, band policy, resolved factors, and per-band weights are persisted.
The Gaussian path is unaffected by non-Gaussian weighting.

Raw `C_L^phiphi` and CAMB's scaled
`[L(L+1)]^2 C_L/(2*pi)` arrays remain distinct. The independent known-answer
fixture keeps its response in raw coordinates while feeding scaled arrays to
the same API used by production. Their ratio is `L`-dependent so convention
mixing cannot hide.

#### Why

Multiplying fractional derivatives by the raw covariance again introduces an
extra `C_L^2`, wrong dimensions, and an enormous scale error. Structural checks
such as symmetry and positive semidefiniteness cannot detect a positive
diagonal reweighting.

#### Acceptance evidence

The owning evidence is the workstation cmb-smoke gate: it runs the real
script on CAMB and requires the six assembled blocks (three per-spectrum,
three cross) to be symmetric, positive semidefinite, and alive off the
diagonal, the stencil step study to converge, and the weighting facts to
appear in the output's provenance record. There is no focused CPU fixture
for this section.

### Covariance config schema

#### Rule

`validate_lcdm_params` runs before Cobaya or CAMB construction: the
fiducial params block must be fixed flat-LCDM, only the allowed parameter
names may appear, and a dark-energy or curvature name refuses with the
allowed set named. A required YAML block that is absent raises a KeyError
naming it; the script reads no silent scientific default. The write side —
the refusal of an existing output, the finite-array requirement, and the
temporary-name rename — is owned by the CMB covariance output section of
`ai/notes/data-generation-and-cuts.md`.

Raw and scaled lensing arrays cover every multipole through the requested
maximum. No zero padding substitutes for absent input. Main requests power
through `max(lmax, lens_lmax)`.

#### Acceptance evidence

The cmb-smoke gate runs the script end to end on a valid config. The
refusal paths — a non-LCDM parameter name, a missing required block, an
occupied output name, a nonfinite computed array — are stated by the
script and currently have no dedicated CPU test.

### Stencil step study

#### Rule

The band derivative uses the 5-point central stencil at every configured
step fraction. The relative spread of the derivative across those steps is
reported per band, and a spread above `converge_rtol` is a loud failure: a
finite-difference derivative is sensitive to its step size, so convergence
across steps is required output, not an optional diagnostic. The
smallest step's derivative is the one used in the assembled covariance.

#### Acceptance evidence

The cmb-smoke gate runs the real script on CAMB with two bands and the
configured step list; the step study and the fractional-amplitude
contraction are asserted in the output's provenance record.

### Noise and covariance positive semidefiniteness

#### Rule

The T/E map-noise amplitudes define a physical two-by-two covariance:

```text
delta_te^2 <= delta_tt * delta_ee
```

`delta_tt` is the configured temperature-map noise amplitude, `delta_ee` is
the E-mode-polarization-map noise amplitude, and `delta_te` is their configured
correlated cross-noise amplitude. All three are read from `cov_args.noise` in
microkelvin-arcminutes. The inequality is the positive-semidefinite condition
for that two-field noise covariance.

The producer checks this inequality using a representation-derived rounding
band. After adding signal, each
multipole's TT/TE/EE block satisfies the corresponding inequality. Each
three-by-three Gaussian joint block and the assembled dense joint covariance
is positive semidefinite within one scale-aware numerical tolerance. Invalid
matrices refuse; they are never clipped, loaded, or absolute-valued.

#### Acceptance evidence

The `1,1,10` noise witness produces a negative joint eigenvalue and refuses.
Equality and just-inside rounding controls pass; one representable step outside
fails. Signal-induced and dense-assembly PSD violations refuse at their owning
boundaries.

### Finite derived arithmetic and publication

#### Rule

The validator proves that beam and noise exponentials, Gaussian variance
products, non-Gaussian blocks, and every output array are finite before
`np.savez`. Bounds are derived in log space from resolved multipole and beam
values. Refusal names the first output key, index, and value. A finite noise
amplitude is not enough when its square or covariance product overflows.

#### Acceptance evidence

The high-multipole wide-beam witness has finite noise but overflowing
covariance and refuses. The exact next-representable finite boundary passes.
A mutation checking only input finiteness reaches publication with Inf and must
fail.

### Fiducial cosmology schema

#### Rule

The covariance `params` mapping is complete and explicit. Every value is a
finite native non-Boolean real. Exactly one amplitude key among `As` and
`logA` is present. Exactly one expansion key among `H0`, `thetastar`, and
`cosmomc_theta` is present. `ns`, `omegabh2`, `omegach2`, `tau`, and `mnu`
are required. `omk`, `w`, and `wa` are explicitly present at the supported
flat-LCDM values.

The validated mapping object itself goes to Cobaya and provenance. Omitted
external defaults may not define the science silently.

The repository's reference control uses:

```yaml
As: 2.1e-9
ns: 0.9660
H0: 67.36
omegabh2: 0.02237
omegach2: 0.1200
tau: 0.0544
mnu: 0.06
omk: 0.0
w: -1.0
wa: 0.0
```

#### Acceptance evidence

Empty mapping, Boolean, NaN, Inf, missing or duplicate amplitude, missing or
duplicate expansion parameter, missing singleton, missing fixed LCDM fact, and
nonflat value all refuse. The valid mapping is object-identical at consumption
and persistence. Entry-point mutation checks prove that config, params,
signal/noise PSD, and final-finiteness validators are all called by `main()`.

### Scope of covariance evidence

Pure CPU schema and arithmetic witnesses do not claim real CAMB execution.
The real-CAMB reference config remains the scientific reasonableness control.
Extreme synthetic inputs prove schema totality or mutation catch power; they do
not by themselves prove that a reasonable reference cosmology has a wrong
science result.

## Dense covariance training is unsupported

### Rule

Emulator training supports the diagonal CMB covariance path only. The separate
CMB covariance generator may produce dense covariance for other scientific
uses; that generator capability does not imply emulator-training support. A
training configuration or saved emulator that requests dense covariance
refuses before model construction.

### Reason

A dense covariance can couple different multipoles, so the supported diagonal
scale cannot represent its scientific metric. Silently using the diagonal
training path would report the wrong chi-squared while still producing finite
results.

### Implementation boundary

The training configuration, geometry, loss, artifact, and rebuild path all
enforce the same diagonal-only boundary. Dense emulator training requires a
separate approved ticket and a deliberate permanent-note change; this section
does not pre-authorize its design.

### Acceptance evidence

A diagonal control passes unchanged. A dense-training request refuses before
model construction, while a dense covariance-generation request remains
available to the separate covariance producer.

## Claims that must remain narrow

- Synthetic adapter stubs do not prove a real Cobaya lifecycle.
- A diagnostics PDF proves output creation, not scientific interpretation.
- Structural covariance checks do not replace an independent numerical known
  answer.
- Pure schema witnesses do not replace a real-CAMB reference run.
- Exact multipole width does not prove multipole identity; the stored axis does.
- Persisted fixed facts receive a basic runtime comparison only when the
  artifact and model expose a concrete constant under the same name. Custom
  renamed or derived parameterizations remain the user's responsibility.
- A process exit or prior log never substitutes for current value assertions.
