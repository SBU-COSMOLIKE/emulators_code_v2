# Models, designs, and scientific constraints

This note records the durable model rules that future changes must preserve.
It is a design specification, not a development diary. Every statement is
written as a durable rule, reason, code owner, or acceptance requirement.

## Vocabulary used throughout this note

HyperText Markup Language (HTML) anchors give sections stable link targets. A
convolutional neural network (CNN) uses shared filters to learn local
structure. YAML is the human-readable settings-file format used by training
configurations. CUDA is NVIDIA's accelerator-computing platform. A graphics
processing unit (GPU) is the accelerator device used by CUDA training checks.

An **artifact** is a saved emulator result containing trained weights and
the facts needed to rebuild them. PyTorch is the tensor and machine-learning
package used by the models; a **tensor** is a numerical array with a fixed
shape and data type. A PyTorch **state dictionary** maps stable names to
parameter tensors and registered buffers. A fixed model **buffer** is
a saved tensor used by the model but not changed by gradient updates. A
**drift check** compares candidate behavior with a declared reference and must
detect a deliberate change.

An **identity** is the saved set of facts or byte fingerprints used to decide
whether two datasets, runs, models, or coordinates are the same; it is not the
mathematical identity matrix.

A model **geometry** owns the mapping between physical outputs and the
coordinates used by a network and loss, including axes, masks, scaling, and
fixed basis changes. To **whiten** a residual is to transform it so its
covariance is the identity; a **basis transform** is the fixed matrix that
moves values between two coordinate representations. A **trunk** is the main
network that produces the initial prediction. A **correction head** is a
smaller branch that adds a structured residual to that prediction.

A multilayer perceptron (MLP) is a sequence of learned linear maps and
nonlinear activations. A **Jacobian** is the matrix of output derivatives with
respect to inputs. In a convolution, a **channel** is one feature sequence, a
**group** restricts which input and output channels mix, a **stride** is the
number of positions advanced between outputs, and the **receptive field** is
the set of input positions that can affect one output. A **bin** is one
physically meaningful group of adjacent output coordinates.

A transformer repeatedly combines attention and a per-token MLP. A **token**
is the feature vector representing one bin or contiguous coordinate segment.
An **attention window** is the segment assigned to one token, and an
**attention head** is one independent set of learned projections that compares
tokens. **Permutation equivariance** means that reordering tokens causes the
same reordering of outputs instead of changing their values. Feature-wise
linear modulation (FiLM) lets parameters produce a scale `gamma` and offset
`beta` for hidden features; identity initialization starts with `gamma = 1`
and `beta = 0`.

A **registry** is a fixed mapping from accepted configuration names to the
classes or functions that own them. A **capability flag** is an explicit class
property that tells shared code which geometry, data, or head behavior the
class supports.

CosmoLike is the cosmological-likelihood calculation that produces the data
vectors used by the cosmic-shear family. A driver **fileroot** is the configured
path stem shared by one run's output files. `n(z)` is a galaxy redshift
distribution. A **gate** is a named validation job whose required result is
written before it starts. The **board runner** is the script that executes
registered gates and records their raw results.

## How to use this note

Every independent technical contract answers four questions:

1. **Rule:** What behavior is required?
2. **Reason:** What scientific or numerical failure does the rule prevent?
3. **Code ownership:** Which module owns the behavior?
4. **Acceptance evidence:** Which exact checks distinguish the required
   behavior from a plausible but wrong implementation?

One long section may answer the four questions once and then list closely
related subrules under descriptive bold labels. Named reference summaries and
tables may organize definitions, ownership, or evidence shared by several
contracts. A new enforceable behavior receives NO-GO when its section leaves
any of the four answers implicit.

The primary code owners are `emulator/designs/`, `emulator/losses/`, and
`emulator/activations.py`. Gate evidence is recorded under stable HTML
anchors so external gate metadata can link to the exact contract.

## The architecture family

**Code ownership.** The model family lives in
`emulator/designs/` and `emulator/losses/`. The principal design modules are
`blocks.py`, `plain.py`, `ia.py`, and `pce.py`; the principal loss modules are
`core.py`, `ia.py`, `pce.py`, `scalar.py`, `cmb.py`, and `transfer.py` inside
those directories. The shared `emulator/activations.py` module remains flat
because code outside this family imports it directly and drift checks patch it
by path. Moves must preserve saved `.emul` state dictionaries, saved
class markers, and the explicit polynomial chaos expansion (PCE) import.

**Rule.** Every model is selected through the shared registry and declares
structural capabilities for geometry, bins, parameters, and correction-head
type. Trunks, correction heads, coordinate attachments, identity
initialization, training phases, and conditioning follow the subrules below.

**Reason.** A parallel construction path or class-name test can build a model
that trains but cannot be staged, saved, rebuilt, or served under the same
scientific coordinate identity.

**Acceptance evidence.** Configuration, initialization, phase, geometry,
save/rebuild, and family gates must exercise every declared capability. A
mutation that bypasses a capability or changes coordinate order must fail its
own named check.

**Registries and capabilities.** Intrinsic alignment (IA) describes the
alignment of galaxy shapes with the surrounding tidal field. The nonlinear
alignment model (NLA) and the tidal-alignment/tidal-torquing model (TATT) are
represented by the shared registry and model infrastructure. Deployable TATT
training additionally requires a validated ten-template dataset; registry
construction alone is not a claim that such a production dataset is present.
The `MODELS` registry uses `(name, ia)` keys, where
`name` is `resmlp`, `rescnn`, or `restrf`, and `ia` is `None`, `nla`, or
`tatt`. `IA_DESIGNS` owns each form's amplitude names, coefficient function,
and template count. New IA forms extend this table rather than creating
parallel model paths. The `factored`, `needs_geom`, `needs_bins`,
`needs_params`, and `head_block` capability flags replace type checks. The
`factored` flag selects the amplitude-aware input geometry and the loss that
combines exact IA templates. The `needs_geom` flag requests the saved output
geometry needed to build fixed coordinate-order buffers. The `needs_bins`
flag requests the per-bin coordinate divisions used by CNN and transformer
heads. The `needs_params` flag tells loss, training, and diagnostic callers
that encoding, decoding, or chi-square evaluation also consumes whitened
model parameters. The `head_block` value is `None`, `"cnn"`, or `"trf"`; it
declares whether the model has a correction head and selects that head's YAML
configuration block. `DesignSpec.__init_subclass__` validates `head_block`
when a model class is defined.

**Trunk and correction head.** A residual multilayer perceptron (ResMLP) trunk
maps cosmological parameters to the target data vector. A small
coordinate-aware convolutional (ResCNN) or transformer (ResTRF) head corrects
the residual that retains angular or spectral structure. A dense trunk cannot
see a permutation of the output coordinate, whereas shared head weights can
use neighboring-coordinate structure with fewer effective degrees of freedom.
Coordinate-aware heads are supported only when a family-specific learning
curve demonstrates a benefit; an isolated comparison never substitutes for
that evidence.

Structured heads operate in physical angular order. `W_fd` maps the fully
whitened basis (`f`) to the diagonal physical-coordinate basis (`d`), and
`W_df` maps back. These transforms are fixed model buffers because a live
geometry call inside `forward()` prevents stable CUDA graph capture.

**Families with coordinate axes.** Correction heads apply to cosmic shear,
cosmic microwave background (CMB), one-dimensional grids, and two-dimensional
grids. A diagonal family is one whose chi-square metric acts independently on
each stored physical coordinate after per-coordinate scaling, without a dense
basis rotation. Diagonal-family geometries already whiten in physical order:
multipole `ell`, redshift `z`, or redshift slices by wavenumber `k`. Their basis
change is the identity, so `W_fd` and `W_df` remain `None`; an explicit square
identity buffer would waste memory. `geometry.attach_head_coords()` supplies
one bin for CMB or a one-dimensional grid and one bin per redshift slice for a
two-dimensional grid. `model.trf.n_tokens` may split a single-bin spectrum
into contiguous attention windows, but multi-bin geometries refuse that
option. Scalar outputs remain trunk-only because named scalar quantities do
not share a coordinate axis.

`emulator/results.py::rebuild_emulator` restores the head-coordinate
attachment. Cosmic-shear artifacts persist `bin_sizes` and `pm_kept`, the per-element flags that
distinguish the xi-plus and xi-minus shear branches, in
`DataVectorGeometry.state()`. Optional attributes remain unset when their
value is `None`, which preserves the `hasattr` capability checks. Rebuild
refuses a structured-head artifact that predates the required geometry fields.

**Identity at initialization.** The last layer of every correction branch is
initialized to zero, so the complete model equals the trunk at construction
and at the two-phase handoff. The condition `a(0) = 0` preserves that identity,
but a trainable branch also needs a finite, representably nonzero derivative
`a'(0)`. ReLU has `a'(0) = 0` and permanently blocks such a branch. The
sign-based power and gated-power formulation has the same exact-zero Jacobian.
The compatibility and refusal contract appears below. `gate_init` is the
initial scalar multiplier applied to the correction branch in
`trunk + gate * correction`. A value of `0.1` gives a soft start, a value of
`1.0` suits a short head phase, and zero creates a deadlock that validation
must refuse.

**Training phases.** Two-phase training belongs to every model with a
correction head, not only an IA model. Plain ResCNN and ResTRF implement
`set_train_phase` for joint, trunk, and head phases. The trunk phase bypasses
the head at pure-trunk cost. The head phase freezes the trunk under
`torch.no_grad()` when `freeze_trunk` keeps its frozen default; setting
`freeze_trunk` false makes the second phase train trunk and head jointly
(`training-stack.md` owns the schedule options). The same trunk-then-head
schedule, `trunk:` and `head:` configuration blocks, and head-activation pin
apply to cosmic shear, CMB, and both grid families. ResMLP is the only
single-phase design.

**Convolutional head.** `pad_idx` scatters each ragged bin into a tensor shaped
`(number of bins, maximum bin length)`. `Conv1d` blocks mix those channels;
the model then gathers physical entries and applies the outer gate.
`kernel_size`, `n_blocks_cnn`, `groups`, `separable`, and `rescale_kernel`
control the design. Group choices must respect probe-block boundaries.
Separable convolution reduces parameter count but is slower on the measured
path. Kernel rescaling grows the receptive field through odd widths. Two
stacked linear convolutions collapse to one linear map, so `act_mid` supplies
the required intermediate nonlinearity.

**Transformer head.** Each token is a raw padded bin segment with natural
width `max_bin`, which is 26 for the Legacy Survey of Space and Time Year-1
(LSST-Y1) configuration. `n_heads` must divide this width.
The design has no embedding or projection adapters, and the per-token MLP
width equals the token width. `mlp_width` requires a new design decision rather
than an ad hoc reintroduction. A shared MLP must preserve the intended
permutation-equivariant behavior.

**Feature-wise conditioning.** Feature-wise linear modulation (FiLM), enabled
by `model.cnn.film` or `model.trf.film`, applies
`gamma(parameters) * hidden + beta(parameters)` in each head block. Identity
initialization makes FiLM a no-op before training while allowing the correction
to depend on cosmology. A factored IA model separates exact amplitude columns
from the emulated input. Its `n_in` value is the number of non-amplitude input
columns, so FiLM must condition only on `x[:, :n_in]`; using appended amplitude
columns would break the exact closed-form factorization. The conditioning
choice still requires a family-specific learning curve; a single trained
configuration cannot establish general performance.

## Activations and norms

**Rule.** Activation families, normalization modes, and correction-head
activation pins accept only the forms and combinations described below.

**Reason.** An activation with the wrong derivative can leave a zero-started
correction branch permanently dead. A normalization or phase-local activation
change can silently alter the model represented by already trained weights.

**Code ownership.** `emulator/activations.py` owns activation mathematics.
The design constructors and experiment validator own normalization and pin
compatibility.

**Acceptance evidence.** The `head-activation-pin` and `relu-tanh-norm` gates
exercise valid construction, exact routing text, incompatibility refusals,
and the activation/norm combinations named by their stable anchors below.

`make_activation` provides six activation families. H is a learnable
interpolation between the identity and Swish, the smooth ramp
`x * sigmoid(x)`, with nonsaturating linear tails.
The other families are power, `multigate(K)` with `K` learned gates,
gated-power, ReLU, and hyperbolic tangent (`tanh`). Learnable shape parameters
are feature-specific. The module-role allowlist defined in
`training-stack.md` keeps weight decay away from those parameters.

`model.norm` accepts `affine`, `per_feature`, or `none`. The default `affine`
form learns one scale and offset per layer. `per_feature` uses
`FeatureAffine` to limit tanh saturation. Batch normalization is excluded
because its batch coupling confounds batch-size and exponential-moving-average
(EMA) experiments. Separate training and evaluation statistics can also bake
the wrong mode into a compiled twin, and batch-normalization buffers are not
part of the EMA parameter average.

`model.cnn.activation` and `model.trf.activation` pin the activation used by a
correction head. The pin is a construction-time, run-wide choice rather than a
phase option because swapping activation families under trained weights would
reinitialize learned shape parameters. A pin is legal only when
`trunk_epochs > 0` and `freeze_trunk` is true. `head: activation:` is an alias;
specifying both spellings causes refusal. `trunk: activation:` is invalid and
must produce a teaching error.

### Gate-evidence vocabulary

The acceptance blocks below use the following terms:

- A **gate** is a named validation job whose required result is written before
  it starts. A **leg** is one named check and supports only the claim stated by
  that leg.
- A **golden comparison** runs the candidate code and a trusted reference, then
  compares the declared observations. The trusted reference is the **pinned
  base**. Its identity is a full Git commit hash, called the **base commit**;
  an abbreviated branch name or moving reference is insufficient.
- **Selected text** is the explicitly chosen subset of process-output lines
  that a leg compares. Equality of empty selections is not evidence.
- A **gate manifest** is the declared list of files and capabilities on which a
  gate depends. A **manifest-bound** input is named directly by that manifest.
  The **manifest hash** is the digest that protects the declared inputs from
  unnoticed changes. The board runner defined at the top of this note
  executes the gate and records its raw log.
- **Transitive reads** are files reached indirectly through a declared pointer
  or configuration file. Unless the manifest also names them, they are not
  protected by the manifest hash.

<a id="head-activation-pin-evidence"></a>
### Acceptance evidence: `head-activation-pin`

**Rule.** The registered gate checks the configured pin through process exit
results and selected startup text. Process and text checks establish option
routing and refusal behavior; they do not establish trained parameter values
or numerical prediction agreement.

- files: reads `ai/gates/configs/head-activation-pin-config.yaml`,
  `ai/gates/configs/head-activation-pin-license.yaml`, and the cosmic-shear
  training/validation arrays, parameter tables, covariance, and CosmoLike
  `.dataset` pointer named by the gate manifest. The driver follows that
  pointer to data-vector, covariance, mask, and n(z) siblings that are
  transitive reads outside the manifest hash. A configured golden leg would
  also read `cosmic_shear_train_emulator.yaml` and stage one temporary copy in
  the configured driver fileroot for both the candidate and pinned drivers.
  Successful training calls write the driver's ordinary `.emul` and `.h5`
  products, but this gate does not read those products back; the board runner
  writes the gate's raw log.
- subprocess: runs `driver/cosmic_shear_train_emulator.py` for the pinned-head
  configuration, for that configuration plus `--activation=power`, and for
  the deliberately invalid unfrozen-head configuration. A golden leg supplied
  with a pinned base runs the candidate and pinned drivers once each. There is no separate
  `ai/gates/checks/` child.
- metric: per-leg.  The executable legs use exact process-exit predicates,
  literal selected-text containment, or a case-insensitive selected-text
  regular expression.  The conditional golden leg compares only selected log
  lines after removing their trailing wall-clock field; it is not a raw-byte
  comparison, and the helper does not require either selected-line list to be
  nonempty.
- legs: 5, named `head-activation-pin.golden-selected-text-equality`,
  `head-activation-pin.pinned-config-exit-zero`,
  `head-activation-pin.multigate-text-present`,
  `head-activation-pin.flag-vs-pin-warning`, and
  `head-activation-pin.unfrozen-pin-refusal`.
- acceptance boundary: four process/text legs are executable when the Torch,
  CosmoLike, and GPU requirements are available. These legs do not assert a
  parameter count even when the output contains related design information.
  Golden selected-text equality requires `board_config.json` to name a
  reviewed pinned base and requires an assertion that both selected-line
  lists are nonempty. Without both requirements, equality cannot prove that
  any selected text existed.
- capability boundary: a manifest-bound GPU run is required. Numerical
  prediction agreement requires an additional trained-artifact comparison;
  startup text cannot supply that evidence.

<a id="head-activation-pin-golden-selected-text-equality"></a>
`head-activation-pin.golden-selected-text-equality` requires a configured base
commit. The leg compares selected candidate/base log-line lists after stripping
the trailing wall-clock value. The helper accepts two empty lists, so the leg
must also assert that both selections are nonempty before the comparison counts
as evidence.

<a id="head-activation-pin-pinned-config-exit-zero"></a>
`head-activation-pin.pinned-config-exit-zero` — the process running the
pinned-head configuration exits with status zero.

<a id="head-activation-pin-multigate-text-present"></a>
`head-activation-pin.multigate-text-present` — the captured output from the
pinned-head configuration contains the literal text `multigate`.

<a id="head-activation-pin-flag-vs-pin-warning"></a>
`head-activation-pin.flag-vs-pin-warning` — the run with
`--activation=power` both exits with status zero and prints that the head keeps
its `multigate` pin.

<a id="head-activation-pin-unfrozen-pin-refusal"></a>
`head-activation-pin.unfrozen-pin-refusal` — the deliberately invalid
unfrozen-head configuration exits nonzero and its captured output contains
`frozen`, matched without regard to letter case.

<a id="relu-tanh-norm-evidence"></a>
### Acceptance evidence: `relu-tanh-norm`

**Rule.** The registered gate pairs `relu` with `per_feature` normalization and
`tanh` with `affine` normalization. A deterministic CPU child tests the real
activation and normalization factories on a small nonlinear regression. The
two full scientific configurations separately prove driver reachability when
the required workstation is available.

- files: reads `ai/gates/configs/relu-tanh-norm-per-feature.yaml`,
  `ai/gates/configs/relu-tanh-norm-affine.yaml`, and the cosmic-shear
  training/validation arrays, parameter tables, covariance, and CosmoLike
  `.dataset` pointer named by the gate manifest. The driver follows that
  pointer to data-vector, covariance, mask, and n(z) siblings that are
  transitive reads outside the manifest hash. A configured golden leg would
  also read `cosmic_shear_train_emulator.yaml` and stage one temporary copy in
  the configured driver fileroot for both the candidate and pinned drivers.
  Successful calls write the driver's ordinary `.emul` and `.h5` products,
  but this gate does not read those products back; the board runner writes the
  gate's raw log.
- subprocess: runs `driver/cosmic_shear_train_emulator.py` once for the
  `relu`/`per_feature` configuration and once for the `tanh`/`affine`
  configuration. The board also runs
  `ai/gates/checks/d5_training_behaviors.py --gate relu-tanh-norm`. A golden
  leg supplied with a pinned base runs the candidate and pinned drivers once
  each.
- metric: per-leg. The CPU child checks exact ReLU/Tanh values, identity
  initialization of the selected norms, finite strict loss descent, and a
  final loss below half the mean-only predictor's loss. The driver legs use
  exact zero-exit predicates and literal selected-text containment. The
  conditional golden leg compares selected log lines after removing their
  trailing wall-clock field; it is not a raw-byte comparison, and both
  selected-line lists must be nonempty.
- legs: 7, named `relu-tanh-norm.golden-selected-text-equality`,
  `relu-tanh-norm.per-feature-config-exit-zero`,
  `relu-tanh-norm.per-feature-text-present`,
  `relu-tanh-norm.affine-config-exit-zero`, and
  `relu-tanh-norm.affine-text-present`, plus
  `relu-tanh-norm.relu-finite-descent` and
  `relu-tanh-norm.tanh-finite-descent`.
- acceptance boundary: the two CPU legs establish factory-level numerical
  behavior on the fixed small regression. They do not claim that the complete
  cosmic-shear jobs ran. The four process/text legs require Torch, CosmoLike,
  and a GPU. Golden selected-text equality additionally requires a reviewed
  pinned base and nonempty selected-line lists.
- capability boundary: the CPU child requires Torch. The separate full-driver
  legs remain workstation evidence and must not be inferred from the CPU
  result.

<a id="relu-tanh-norm-golden-selected-text-equality"></a>
`relu-tanh-norm.golden-selected-text-equality` requires a configured base
commit. The leg compares selected candidate/base log lines after stripping the
trailing wall-clock value. The helper accepts two empty lists, so the leg must
also assert that both selections are nonempty before equality counts as
evidence.

<a id="relu-tanh-norm-per-feature-config-exit-zero"></a>
`relu-tanh-norm.per-feature-config-exit-zero` — the process whose YAML requests
`relu` with `per_feature` normalization exits with status zero.

<a id="relu-tanh-norm-per-feature-text-present"></a>
`relu-tanh-norm.per-feature-text-present` — that process's captured output
contains the literal text `per_feature`.

<a id="relu-tanh-norm-affine-config-exit-zero"></a>
`relu-tanh-norm.affine-config-exit-zero` — the process whose YAML requests
`tanh` with `affine` normalization exits with status zero.

<a id="relu-tanh-norm-affine-text-present"></a>
`relu-tanh-norm.affine-text-present` — that process's captured output contains
the literal text `affine`.

<a id="relu-tanh-norm-relu-finite-descent"></a>
`relu-tanh-norm.relu-finite-descent` — the production ReLU and per-feature
normalization factories have their exact initial behavior, and their fixed
small regression finishes with a finite loss below both its initial loss and
half the mean-only loss. A dead network and a mean-only result fail.

<a id="relu-tanh-norm-tanh-finite-descent"></a>
`relu-tanh-norm.tanh-finite-descent` — the production Tanh and affine
normalization factories have their exact initial behavior, and their fixed
small regression finishes with a finite loss below both its initial loss and
half the mean-only loss. A dead network and a mean-only result fail.

## Factored IA (what "factored" means)

**Rule.** Factoring removes IA parameters that enter the data vector as exact
polynomial coefficients from the neural-network input. NLA uses the amplitude
`A1` and three templates with coefficients `[1, A1, A1**2]`. TATT uses the
amplitudes `a1`, `a2`, and `b_TA` and ten polynomial templates.
`AmplitudeFactorGeometry` appends those raw amplitudes after the first `n_in`
non-amplitude columns. The loss reads each sample's amplitudes and combines the
templates in closed form. Redshift-evolution powers controlled by `eta` do not
factor and remain emulated.

**Reason.** The construction uses existing scattered samples without a new
simulation or an artificial division of the dataset. A wider amplitude prior
gives factoring more leverage; narrow NLA priors may remain neutral, whereas
the coupled TATT amplitudes provide the intended use case. An exact written
parameter dependence must not be replaced by a learned approximation.

**Code ownership.** `emulator/geometries/parameter.py` owns amplitude-column
placement. `emulator/designs/ia.py` and `emulator/losses/ia.py` own the
factored network and exact template combination. `emulator/experiment.py`
owns configuration and staging compatibility.

**Acceptance evidence.** Family identity gates require exact template
coefficients, amplitude-column order, epoch-zero composition, save/rebuild
identity, and refusal of an incompatible parameter or family declaration. A
mutation that asks the network to learn an exact amplitude coefficient must
fail. TATT is advertised for production only when a real ten-template dump
passes the same checks.

## NPCE (the pce: block)

**Rule.** NPCE fits a validated polynomial base once and trains a neural
residual or ratio only where the family metric supports that form. The saved
artifact records enough base and decoder state to reproduce composition after
rebuild.

**Reason.** Refitting the base inside a sweep, combining it in the wrong
coordinate space, or omitting it during inference changes the mathematical
model while leaving network shapes plausible.

**Code ownership.** `emulator/designs/pce.py` owns the polynomial basis,
leave-one-out selection, fitted coefficients, and calibrated input domain.
`emulator/losses/pce.py` owns composition and target encoding.
`emulator/experiment.py` owns fitting and configuration validation.
`emulator/inference.py` owns rebuilt prediction.

**Acceptance evidence.** The `npce-training` gate and family identity gates
cover base fitting, supported forms, refusals, distinct sweep fits,
save/rebuild composition, and a mutation that omits the base contribution.

A neural model with a PCE base is called NPCE in the configuration and code.
The top-level `pce:` block uses these values:

- `form` is `residual` or `ratio` and states how the neural correction combines
  with the polynomial base.
- `p_max` bounds the polynomial degree under the hyperbolic basis rule and is
  the main smoothness limit. The hyperbolic rule keeps a term whose
  per-parameter degrees are `a_1 ... a_m` only when
  `(sum_i a_i^q)^(1/q) <= p_max`; for `q < 1` this prunes terms that spread
  degree across several parameters before terms that concentrate it in one.
- `r_max` is the largest number of input parameters allowed to interact in one
  polynomial term.
- `q`, in `(0, 1]`, is the hyperbolic sparsity exponent. Smaller values penalize
  terms that spread degree across several parameters more strongly.
- `k_max` is the maximum number of leading singular-value-decomposition (SVD)
  output modes to try.
- `loo_max` is the maximum relative leave-one-out (LOO) error for retaining an
  output mode.
- `max_terms` caps the active polynomial basis terms fitted for each output
  mode.
- `max_fail` stops the output-mode search after that many consecutive modes
  fail the LOO requirement.

The block stays outside `train_args` because
`sweep_hyperparam` stages the base once; sweeping a PCE option there would
change the option without refitting the base. PCE is mutually exclusive with
rescaling and factored IA.

A PCE base adds capacity rather than replacing the neural model because the
shape modes are not low-degree polynomials. NPCE is supported infrastructure,
not an established way to lower the sample-efficiency floor. The fit uses low
degree to limit Runge oscillation, retains only modes with acceptable
leave-one-out (LOO) error near the default `loo_max` of 0.05, stops early, and
uses a greedy residual-correlation search on the CPU with the closed-form
predicted-residual-sum-of-squares (PRESS) form of the leave-one-out (LOO)
error. The function keeps its public `select_lars_loo` name, but it does not
execute the least-angle-regression path algorithm.

Scalar, CMB, one-dimensional grid, and two-dimensional grid families wrap
`emulator/losses/pce.py::PCEResidualDiagChi2`, a subclass of
`CmbDiagonalChi2`. Their diagonal metric is the family chi-square, and a
roughness term composes with it because prediction minus target is the complete
whitened residual.
`emulator/experiment.py::EmulatorExperiment._fit_diag_pce` owns fitting across
the four `build_geometry` branches. Every inference branch uses
`emulator/inference.py::_build_diag_decoder`; a bare `geometry.decode` would
omit the saved base contribution.

Diagonal families support residual form only because ratio form depends on a
dense covariance; `validate_pce(diagonal=True)` owns that refusal. CMB permits
NPCE only with `amplitude_law: none` because the amplitude-law loss owns target
construction. A cosmic-shear conclusion does not transfer automatically to
the matter-power-spectrum family; its PCE fits the boost in the nonlinear
matter-power regime. That boost is
`B(k, z) = P_nonlinear(k, z) / P_linear(k, z)`, rather than either power
spectrum by itself.

<a id="npce-training-evidence"></a>
### Acceptance evidence: `npce-training`

**Rule.** The registered gate checks process results and selected NPCE text for
residual, ratio, refusal, and two-point-sweep configurations. Its smoke helpers
establish routing and refusal behavior only. Loss comparison and a distinct
base fit inside each sweep worker require separate executable witnesses.

- files: reads the five `ai/gates/configs/npce-training-*.yaml` training and
  refusal configurations, the cosmic-shear training/validation arrays,
  parameter tables, covariance, and CosmoLike `.dataset` pointer named by the
  gate manifest. The driver follows that pointer to data-vector, covariance,
  mask, and n(z) siblings that are transitive reads outside the manifest hash.
  A configured golden leg would also read
  `cosmic_shear_train_emulator.yaml` and stage one temporary copy in the
  configured driver fileroot for both the candidate and pinned drivers.
  Successful calls write the drivers' ordinary `.emul`/`.h5` and sweep
  products; this gate does not read a saved NPCE artifact back, and the board
  runner writes the gate's raw log.
- subprocess: runs `driver/cosmic_shear_train_emulator.py` for residual and ratio
  NPCE, for the invalid NPCE-plus-IA configuration, and for NPCE plus the
  `--rescale=residual` flag.  It runs
  `driver/cosmic_shear_sweep_ntrain_emulator.py` with a requested two-point training
  set-size grid. A golden leg supplied with a pinned base additionally runs the
  candidate and pinned single-training drivers. There is no separate `ai/gates/checks/`
  child.
- metric: per-leg.  The executable legs use exact process-exit predicates,
  literal or regular-expression selected-text checks. The sweep requires
  exactly one result for each requested size, the exact configured threshold,
  and a finite fraction in `[0, 1]`, plus a staging-banner check. An explicit
  worker-failure line, malformed result, duplicate size, missing size, or
  unexpected size fails the leg. The conditional golden leg compares selected
  log lines after removing their trailing wall-clock field; it is not a
  raw-byte comparison, and the helper does not require either selected-line
  list to be nonempty.
- legs: 9, named `npce-training.golden-selected-text-equality`,
  `npce-training.residual-config-exit-zero`,
  `npce-training.residual-pce-text-present`,
  `npce-training.ratio-config-exit-zero`,
  `npce-training.ratio-pce-text-present`,
  `npce-training.pce-ia-refusal`,
  `npce-training.pce-rescale-refusal`,
  `npce-training.sweep-result-lines-and-pce-banner`, and
  `npce-training.rebuild-vs-base`.
- acceptance boundary: seven process/text legs are executable when the Torch,
  CosmoLike, and GPU requirements are available. Golden selected-text equality
  requires `board_config.json` to name a reviewed pinned base and requires
  nonempty selected-line lists. A printed rebuild instruction is process text,
  not a rebuild-versus-base comparison.
- capability boundary: a manifest-bound GPU run is required. Numerical loss
  descent requires a numerical assertion. Per-worker NPCE refitting requires
  worker-specific fit evidence. Saved-artifact equivalence requires an
  executable rebuild-versus-base comparison. None of these properties can be
  inferred from process completion or selected startup text.

<a id="npce-training-golden-selected-text-equality"></a>
`npce-training.golden-selected-text-equality` requires a configured base
commit. The leg compares selected candidate/base log lines after stripping the
trailing wall-clock value. The helper accepts two empty lists, so both
selections must be asserted nonempty before equality counts as evidence.

<a id="npce-training-residual-config-exit-zero"></a>
`npce-training.residual-config-exit-zero` — the residual-form NPCE process
exits with status zero.

<a id="npce-training-residual-pce-text-present"></a>
`npce-training.residual-pce-text-present` — the residual-form process's
captured output contains the literal text `pce`.

<a id="npce-training-ratio-config-exit-zero"></a>
`npce-training.ratio-config-exit-zero` — the ratio-form NPCE process exits with
status zero.

<a id="npce-training-ratio-pce-text-present"></a>
`npce-training.ratio-pce-text-present` — the ratio-form process's captured
output contains the literal text `pce`.

<a id="npce-training-pce-ia-refusal"></a>
`npce-training.pce-ia-refusal` — the NPCE-plus-IA process exits nonzero and its
captured output contains `exclusive`, matched without regard to letter case.

<a id="npce-training-pce-rescale-refusal"></a>
`npce-training.pce-rescale-refusal` — the NPCE process launched with
`--rescale=residual` exits nonzero and its captured output contains
`exclusive`, matched without regard to letter case.

<a id="npce-training-sweep-result-lines-and-pce-banner"></a>
`npce-training.sweep-result-lines-and-pce-banner` — the requested two-point
sweep exits with status zero, prints exactly one finite result in `[0, 1]` for
each of training sizes one thousand and two thousand at the exact threshold
`f(>0.2)`, and prints a line beginning `pce: form`.

<a id="npce-training-rebuild-vs-base"></a>
`npce-training.rebuild-vs-base` requires the wrapper to run and compare a saved
artifact, its rebuilt model, and the pinned base. Printing the comparison
instruction does not establish artifact equivalence.

### NPCE LOO selection is strict

**Rule.** Every retained PCE mode must have finite leave-one-out error below
`loo_max`. No-mode selection refuses the fit instead of retaining a fallback.
Support indices are unique, and selection stops when every usable candidate is
active. The score is strict: equality with `loo_max` does not pass.

Final acceptance describes the base that will actually be saved. The design
uses the saved float32 input bounds and converts training inputs to the same
float32 format used while serving. Coefficients are rounded to float32 before
scoring, and residual prediction uses the same dense float32 matrix
multiplication as serving. After retained modes are assembled, the complete
multi-column coefficient matrix is scored jointly. A joint failure is removed
and the narrower matrix is checked again. The fit refuses only if no mode
remains.

**Reason.** A no-mode fallback in
`emulator/designs/pce.py::PCEEmulator.from_training` could retain mode zero
when no mode satisfies `loo < loo_max`. A persisted base could then contain an
error far above the requested ceiling while the report claims that the
predicate held. In `emulator/designs/pce.py::select_lars_loo`, an all-active
candidate set can leave every score at `-1`; another `argmax` could select
column zero and append a duplicate support index. Either behavior invalidates
the claim that the PCE base passed its selection rule.

**Implementation boundary.** The no-mode fallback is forbidden. A refusal
names the best attempted LOO, the threshold, and the modes tried. Every
recorded or retained LOO must be finite. `X_white` and `Y_white` must be
finite, two dimensional, row-aligned, nonempty in width, and large enough for
the fit. Raw target variance must be finite and positive; an epsilon may not
turn a constant mode into evidence. Leverage must be finite in `[0, 1)` and
may not be clipped to manufacture a denominator.

`select_lars_loo` stops when all usable candidates are active, ignores an
unused exact-zero candidate column, caps active terms at the usable candidate
count and at `n_samples - 1`, and never duplicates support. The constant
candidate must remain usable. `best_beta` and support must exist, align, and be
finite before return. The fit report derives from retained modes that satisfy
the final joint saved-format predicate. `pce.loo_max` requires an explicit
finiteness check because a comparison with NaN does not enforce positivity.

The PCE artifact state remains the six arrays `lo`, `hi`, `multi_index`, `C`,
`Vk`, and `Ybar`; it does not persist historical LOO scores. New fits are
certified before publication, and their resolved PCE configuration records the
requested threshold. An older six-array artifact cannot be retroactively
certified from those arrays alone and should be retrained when its origin
predates strict saved-format acceptance.

**Code ownership.** `emulator/designs/pce.py::PCEEmulator.from_training` owns
the retained-mode decision and calls
`emulator/designs/pce.py::select_lars_loo`, which owns support selection and
its termination rule. `emulator/experiment.py::validate_pce` owns the
configuration value, including finite `loo_max` validation.

**Acceptance evidence.** A predictable control retains a real mode with every
LOO below threshold. A strict-threshold fixture refuses with `no mode passed`
and writes no artifact. NaN or infinity in inputs, targets, LOO values, or
`loo_max` causes refusal. `max_terms > n_candidates` terminates with unique
support, an active count never reaches the number of training rows, and one-
or two-column candidate sets cannot duplicate index zero. A large-offset input
witness proves that pre-round bounds cannot certify a different saved design.
Large-coefficient witnesses prove that promoted float64 multiplication and
separate per-mode products cannot hide float32 cancellation. When one mode
fails only in the final joint matrix, that mode is removed, the narrower matrix
is rechecked, and a different passing mode remains. A valid PCE preserves the
six-array state and real artifact save/rebuild behavior. The training-size
sweep requires exactly one finite result for each requested size. Production
NPCE training requires this complete acceptance set.

## Composition spine

**Rule.** `CosmolikeChi2` stores a geometry object rather than inheriting from
a geometry class. The program builds that geometry once and wraps it with the
selected loss. Loss wrappers forward `dest_idx`, `total_size`, `encode`, and
`decode`. The `needs_params` capability means that encoding, decoding,
chi-square, or loss evaluation consumes whitened parameters in addition to a
prediction. Every diagnostic branches on this capability.

**Reason.** A hard-coded `geometry.decode(prediction)` call bypasses the loss
composition and is wrong for a parameter-aware loss even when array shapes
match.

**Code ownership.** Geometry classes own coordinate state. Loss wrappers own
composition. `emulator/experiment.py`, `emulator/training.py`, and
`emulator/diagnostics.py` consume the declared capabilities.

**Acceptance evidence.** Each parameter-aware loss must pass encode/decode,
chi-square, diagnostic, and save/rebuild checks through the wrapper. A
mutation that calls the bare geometry decoder must fail on a nonzero
parameter-dependent fixture.

## Model configuration values are validated before construction

**Rule.** Values in the selected model block keep the type and meaning written
in YAML. The first check runs before facts files, training arrays, a saved
source, an accelerator, or learned layers are touched. A second check uses the
resolved output geometry before learned layers are constructed. A shared YAML
may still contain settings for an unused CNN or Transformer alternative; only
the selected architecture is checked.

**Reason.** The name map in `emulator/experiment.py::MODEL_BLOCK_KEYS` cannot
by itself validate the mapped values before
`EmulatorExperiment.build_specs` passes them to design constructors. Malformed
values can silently demote the requested architecture instead of producing a
clear refusal:

- A value of `0` for `model.trf.n_blocks` builds an empty transformer-block
  list. `emulator/designs/ia.py::TemplateResTRF.forward` then leaves `t == t0`,
  so the correction `corr = t - t0` remains zero. The trunk can still train
  and pass aggregate collapse checks even though the requested transformer
  head performs no work. Acceptance evidence must therefore prove that a
  requested head cannot silently reduce to its trunk.
- A quoted value such as `"false"` is a nonempty string and is therefore
  truthy in Python. Passing `rescale_kernel`, `separable`, `film`, or
  `shared_mlp` to a constructor without type validation can incorrectly
  enable the corresponding design option.
- A value of `0` for `model.cnn.n_blocks` reaches `self.convs[-1]` in
  `emulator/designs/ia.py::TemplateResCNN.__init__` and raises an unrelated
  `IndexError`. A value of `0` for `model.trf.n_mlp_blocks` similarly reaches
  `self.mlp_lins[-1]` in
  `emulator/designs/blocks.py::TRFBlock.__init__`.
- A value of `0` for `n_heads` causes the expression `dim % n_heads` to
  divide by zero. An incompatible positive value is guarded by a Python
  assertion, which is removed under `python -O`. Public configuration checks
  for `kernel_size`, `groups`, and geometry assumptions must likewise use
  typed exceptions rather than assertions.
- The constructor converts `gate_init` with `float()`, which would otherwise
  accept NaN or infinity. Because the correction starts at zero, the
  expression `out = y + gate * corr` can then produce NaN immediately.
- Converting `n_gates` or `n_tokens` with `int()` would accept booleans,
  truncate fractional values, and accept numeric strings. It could also send
  zero to code that allocates an empty gate tensor. Validation must preserve
  the declared type instead of coercing these values.
- A zero-initialized correction layer initially sends zero into its activation.
  `relu` has zero derivative there, so it cannot wake the entire requested CNN
  or Transformer head. `H`, `multigate`, `tanh`, and the power families
  (`power`, `gated_power`, whose signed power transform is an even magnitude
  ratio with analytic origin derivative one) are live at that starting point.
  ReLU is still a valid activation inside an MLP trunk.

**Implementation boundary.** One pure active-model value validator runs twice.
`EmulatorExperiment.from_config` calls it after selecting the model class and
before files, devices, sources, or construction. `build_specs` repeats it for
values produced by a parameter search. The second call has the output geometry
and can therefore check physical CNN grouping, Transformer token width, and
attention-head divisibility before translating the values into constructor
arguments.

Boolean fields require YAML booleans without truthiness or coercion. Integral
fields reject booleans, strings, and fractional values. Width, gate count,
head depth, Transformer MLP depth, and attention-head count are positive. An
MLP trunk depth of zero remains valid because it explicitly requests the
documented linear-only trunk. `n_tokens` is `None` or an exact positive integer
followed by geometry-dependent checks. `kernel_size` is positive and odd.
`groups` is an exact allowed value for the selected design. `gate_init` is a
finite, real, non-Boolean value that remains nonzero when stored as `float32`.
`model.norm` accepts `affine`, `per_feature`, or `none`.
`model.compile_mode` accepts `default`, `reduce-overhead`, or YAML `null`.
Typed exceptions preserve these checks under ordinary and optimized Python.

**Code ownership.** Shared exact-value helpers live in
`emulator/validation.py`. Model selection, the two validator calls, key
translation, and head-activation parsing live in `emulator/experiment.py`.
Constructors in `emulator/designs/plain.py`, `emulator/designs/ia.py`, and
`emulator/designs/blocks.py` repeat their local checks before allocating
learned layers so direct internal calls cannot bypass the public path.

**Acceptance evidence.** `ai/tests/test_active_model_validation.py` compares
real and quoted Booleans, exact and coercible counts, valid and invalid gates,
head depths, kernels, groups, token layouts, attention widths, normalization,
compile modes, and head activations. Errors name the full dotted setting. One
public `from_config` example proves refusal before file, device, source, or
model access. One `build_specs` example proves that a searched attention value
is checked against the built geometry before spec translation. Direct
constructors run in ordinary and optimized Python. Small valid CNN and
Transformer examples prove that the requested blocks exist and that the final
zero-initialized correction layer receives a finite nonzero gradient and
changes on its first optimizer step.

### Transformer token width must be at least two

**Rule.** Every transformer token has width of at least two. The active-model
validator derives token widths from the physical geometry and refuses a
configuration whose maximum token width is below two.

**Reason.** `emulator/designs/plain.py::ResTRF.__init__` accepts
`model.trf.n_tokens` from two through the full output length. Setting
`n_tokens == n_out` creates one scalar coordinate per token, so
`max_bin == token_width == 1`. The divisibility check accepts `n_heads: 1`
in `emulator/designs/blocks.py::TRFBlock.__init__`, and
`emulator/experiment.py::MODEL_BLOCK_KEYS` together with
`EmulatorExperiment.build_specs` exposes this configuration.

For feature width one, LayerNorm is algebraically input-independent: its mean
is the scalar itself, its variance is zero, and every normalized value is
zero before the learned affine bias. Both pre-normalized branches in
`emulator/designs/blocks.py::TRFBlock.forward` therefore discard the input.
With `film: false`, all
attention and MLP branch outputs are learned constants per token, independent
of cosmology. For any trained weights, `TRFBlock(x)-x` is independent of `x`;
stacking blocks preserves only an input-independent additive correction.
`emulator/designs/plain.py::ResTRF.forward` returns `t - t0` as the head
correction, so the
requested transformer can never learn a sample-dependent correction while
the ResMLP trunk can still train and satisfy aggregate collapse thresholds.
The result is a silent architecture demotion that range and divisibility
checks cannot detect.

**Implementation boundary.** The refusal names the output length, token count,
resolved width, and LayerNorm degeneracy. Plain and factored TRF constructors
share the invariant. Adjacent accepted configurations remain unchanged; no
padding or artificial embedding silently substitutes a different design.

**Acceptance evidence.** A registered Torch leg on a GPU-capable environment
uses a single-bin `N=4, n_tokens=4, n_heads=1, film=false` configuration and
requires refusal before model construction. A control that bypasses validation
and uses deterministic nonzero head weights must produce identical corrections
for two distinct `t0` rows and a zero correction Jacobian with respect to
`t0`. The adjacent `n_tokens=3` configuration must construct and produce an
input-dependent correction. Plain and factored paths must agree. A mutation
that restores only the range and divisibility checks must construct but fail
the behavioral witness.

#### Implementation shape and acceptance evidence

The plain path uses `emulator/designs/plain.py` and
`emulator/designs/blocks.py`. The factored constructor is
`emulator/designs/ia.py::TemplateResTRF`. A guard inside `TRFBlock` catches an
invalid width only after `TemplateResTRF` has allocated its template trunk.
Both model paths must refuse before allocating any learnable layer.
`TemplateResTRF` therefore calls the shared validator, and its existing
bin-size calculation occurs before trunk construction.

`emulator/designs/blocks.py::validate_trf_token_width` owns the rule and its
teaching error. `ResTRF` and `TemplateResTRF` supply the resolved physical
output length, token count, and maximum token width before building their
trunks. `TRFBlock` repeats the check as defense in depth for direct
construction. Accepted widths follow the existing constructors without a new
embedding, padding rule, or projection.

The CPU companion `ai/tests/test_trf_token_width.py` must prove early refusal
under ordinary and optimized Python for the plain and factored models. A
mutation that removes only the factored model-level call must allocate trunk
layers before the block-level guard raises. A width-one control that bypasses
validation must produce identical corrections and a zero correction gradient
for distinct inputs. Adjacent width-two plain, factored, and direct-block
controls must construct.
The `cmb_identity.py` check must retain the existing width-20 ResTRF save and
rebuild leg. Local CPU checks do not replace the registered integration leg.

## The science doctrine

**Rule.** A model proposal must preserve the measured regime, sample count,
validation uncertainty, and physical parameter meaning behind every claimed
benefit. The bold summaries below state the accepted evidence boundaries.

**Reason.** A lower error at one temperature, family, seed, or training size
does not establish a general architectural advantage. Removing that context
can turn a real measurement into a scientifically false design rule.

**Code ownership.** `emulator/designs/` owns architecture, while
`emulator/training.py` and the study drivers own the learning-curve and
validation measurements used to choose it. Data-coverage changes remain owned
by the generation and staging modules.

**Acceptance evidence.** A proposed design change supplies the complete
learning curve, repeated seeds, uncertainty, matched data and training
settings, and the family-specific scientific metric. One trained checkpoint
or one aggregate score is not enough.

- **Sample efficiency is the objective.** `N_train` is the number of training
  samples. `f(Delta chi2 > 0.2)` is the fraction of validation samples whose
  chi-square error exceeds 0.2. The learning curve plots that fraction against
  `N_train`, and `N_target` is the smallest training size with a fraction below
  0.10. The demanding regime combines high sampling temperature `T`, the
  time-varying dark-energy parameters `w0` and `wa`, and TATT intrinsic
  alignment. At `T=16`, increasing the training set from 10,000 to 46,000
  changes the measured fraction from 0.219 to 0.100, which supports a
  data-coverage limitation. Any capacity law requires the complete learning
  curve rather than one training size.
- **Effective dimension determines cost.** Nonlinear parameter dependence,
  rather than nominal parameter count, controls sample demand. Photometric
  redshift shifts and factored IA amplitudes contribute little effective
  dimension. A data-limited floor can be addressed through physical structure,
  such as factoring or informative features, or through point placement, such
  as importance sampling. When failures are diffuse, representation changes
  require evaluation before additional sampling.
- **Small-scale structure determines hardness.** The Hubble constant `H0`
  leads the measured hardness direction. The logarithm of the baryon density
  is negatively correlated with difficulty, so more baryons correspond to an
  easier regime. The physical baryon density `omega_b h^2` remains useful for
  defining a cut, but its sign must not be reused as the hardness gradient.
- **Certification needs statistical margin.** With roughly 400 validation
  samples, binomial uncertainty in a fraction near 0.1 is about 0.015.
  Selecting the best epoch biases the reported fraction downward. Certification
  therefore targets a margin near 0.085 and repeats across seeds with a larger
  validation set.
- **Benchmark evidence must keep its regime.** For `T=256` and 250,000 training
  samples, measured fractions are 0.1558 for ResMLP, 0.1472 for factored NLA,
  and 0.1105 for two-phase ResCNN with factored NLA. These values do not
  transfer to another temperature, family, or training size.

## Designs that evidence does not support

**Rule.** The designs below remain outside the accepted model family unless a
new family-specific experiment satisfies the science-doctrine evidence.

**Reason.** Each design either failed to improve the declared metric, changed
the scientific objective, or was measured only in a regime too narrow to
support adoption.

**Code ownership.** The relevant design, loss, experiment, or sampling module
owns any future implementation. No fallback path may introduce one of these
designs implicitly.

**Acceptance evidence.** Adoption requires matched controls, complete learning
curves, repeated seeds, and a discriminating family gate. A configuration
that merely constructs or trains is not acceptance evidence.

- Scaling NLA factoring by the primordial scalar amplitude `A_s` creates
  errors aligned with the `A_s` direction and is not an accepted design.
- Analytic target rescaling does not improve sample efficiency. The factor
  `R` is the per-output ratio of a fast analytic shear prediction at the
  reference cosmology to the same prediction at the sampled cosmology. Its
  owning formula is `emulator/analytics.py::_analytic_R`. This machinery
  remains optional preprocessing, not a required target transformation.
- A separate dense MLP for each bin discards the shared parameter-to-data map.
  Mean-squared error is not equivalent to chi-square after block whitening, so
  the loss retains the full inverse-covariance contraction.
- A global CNN head is neutral at `T=16`; a benefit measured at high
  temperature must not be generalized to that regime.
- Convolution represented as matrix multiplication exposes a CPU-specific
  performance problem and is not a general replacement.
- `ParallelResMLP`, `template_mix`, gated linear-unit mixing, max-pooling
  heads, batch normalization, a separate transformer `mlp_width`, and
  smoothness priors lack evidence for adoption. Chi-square behaves as a
  high-pass filter whose blind spot is a smooth common mode, so a smoothness
  prior changes the objective rather than repairing it.
- Loss shaping, log-whitened inputs, space-filling sampling, and a local-linear
  floor do not establish a lower sample-efficiency floor. Space-filling
  sampling also conflicts with the deliberate tempered-Gaussian distribution.
- Width beyond 256 shows saturation. A width comparison in which 128 is
  slightly under-capacity and 256 reaches 0.212 cannot establish a width-128
  advantage; capacity comparisons use width 256 as the baseline.

## Recurring gotchas

**Rule.** The implementation invariants below apply whenever their named
model, training phase, compiled path, or diagnostic capability changes.

**Reason.** Each invariant prevents a plausible implementation that remains
finite and shape-correct while disabling learning, changing coordinates, or
reporting the wrong state.

**Code ownership.** The named activation, design, training, geometry, and
diagnostic modules own their respective invariant.

**Acceptance evidence.** The affected change must retain one valid control
and one targeted mutation for each touched invariant. A broad smoke result
does not replace the targeted check.

- Stacked convolutions require an intermediate activation; otherwise two
  linear convolutions collapse to one linear map.
- Basis transforms remain fixed buffers. A live geometry call inside
  `forward()` breaks compiled execution assumptions.
- Scalars used inside a compiled loop remain zero-dimensional tensors on the
  active device.
- Learning-rate warmup updates at the start of an epoch so every batch sees the
  intended rate.
- The epoch-zero baseline evaluation initializes best-model tracking before
  training changes the weights.
- Diagnostics branch on the `needs_params` capability because parameter-aware
  losses cannot be decoded through a parameter-free path.
- Benchmark conclusions remain tied to the measured device and do not transfer
  automatically between CPU, CUDA, and Apple Metal Performance Shaders (MPS)
  hardware.

## Power activations require the analytic derivative at zero

**Rule.** `PowerGatedActivation` and `GatedPowerActivation` preserve both the
claimed forward value and the analytic input derivative at zero. The power
component has derivative one at the origin, including the `p=1` identity
case.

**Reason.** The sign-based expression
`torch.sign(x) * ((1.0 + abs(x)) ** p - 1.0) / p` has the right value but the
wrong Jacobian at `x=0`. Both `sign(x)` and `abs(x)` have zero derivative there,
and the inner magnitude is zero, so automatic differentiation returns a zero
power-component derivative. Identity-initialized correction layers, padding,
and new channels deliberately create exact zeros. A forward identity check
cannot detect this gradient-absorbing point.

**Implementation boundary.** Express the power component as `x` times an even
magnitude ratio,
`psi_p(x) = x * ((1 + abs(x))**p - 1) / (p * abs(x))`, with the analytic
limit one at zero. Compute the near-zero ratio with `log1p` and `expm1` or a
justified series; an unguarded `0/0` branch is forbidden. Constructors require
finite positive `p_min < p_max`, so malformed direct calls cannot drive the
denominator toward zero. Forward values away from the near-zero neighborhood
remain unchanged. Documentation may state the derivative only after executable
evidence proves it.

**Code ownership.** `emulator/activations.py::PowerGatedActivation` and
`GatedPowerActivation` own the forward formulas and trainable power bounds.
`emulator/activations.py::make_activation` owns selection by configuration
name. Head-compatibility validation remains with the active-model validator.

**Acceptance evidence.** A registered Torch leg on a GPU-capable environment
compares H, power, and gated-power values and input gradients at
`x = [-epsilon, 0, +epsilon]` under default initialization. With `p=1`, the
power component equals `x` and has derivative one, including at exactly zero.
A zero preactivation inside a small residual block transmits a nonzero
gradient. Float64 `gradcheck` covers several learned powers. A mutation that
restores `sign(x) * f(abs(x))` must fail specifically at the zero-Jacobian
assertion.

## NPCE refuses queries outside its calibrated domain

**Rule.** A polynomial-chaos-expansion (PCE) base accepts a query only inside
its persisted calibration domain, apart from an explicitly defined
floating-point tolerance. Training, validation, and rebuilt-artifact inference
apply the same domain policy.

**Reason.** `PCEEmulator.forward` maps whitened inputs to a fitted Legendre box
and clamps every mapped coordinate to `[-1, 1]` in
`emulator/designs/pce.py::PCEEmulator.forward`. An input one rounding unit
beyond the boundary and
an arbitrarily distant cosmology can therefore map to the same boundary point.
The output remains finite even though the base evaluated a different
cosmology. A neural PCE residual refiner sees the original input, but its
training target includes the base's saturation; no rule guarantees correction
of an arbitrarily clipped base. Persisted `lo` and `hi` already define the
needed calibration data.

**Implementation boundary.** Persist a named PCE domain policy. Scientific
serving refuses points outside the calibrated whitened box, with only a
documented scale-aware tolerance for floating-point roundoff. Optional clipping
inside that tolerance never substitutes for validation. A refusal names the
stored parameter coordinate, whitened value, allowed `[lo, hi]` interval, and
overshoot, mapped to the input-geometry record when available. Persisted bounds
must be finite, one dimensional, aligned with the PCE input width, and satisfy
strict `lo < hi`. The resolved fit and evaluation record counts exact boundary
hits and near-tolerance points. Leave-one-out selection judges polynomial
quality inside the domain; the domain policy decides whether a query belongs
there.

**Code ownership.** `emulator/designs/pce.py::PCEEmulator.from_training` owns
the calibrated bounds and saved policy. `PCEEmulator.forward` owns the check
before basis evaluation. `emulator/results.py::rebuild_emulator` and
`emulator/inference.py` must preserve and apply the same saved policy after
rebuild.

**Acceptance evidence.** Two far-out inputs on the same side of a boundary
must both refuse rather than collide. Checks cover values below `lo` and above
`hi` in each dimension, nonfinite bounds, equal bounds, shape mismatch, exact
endpoints, and a one-unit-in-the-last-place tolerance control. Training and
rebuilt-artifact inference must agree. The witness covers residual NPCE on one
diagonal family and one dense-covariance family.

### A zero `gate_init` is an absorbing dead head

**Rule.** `gate_init` must be finite, real, non-boolean, and representably
nonzero after conversion to the parameter dtype. Positive sign is not required;
a negative gate is equivalent up to the correction sign.

**Reason.** A structured head returns
`trunk + gate * correction`, and its correction branch starts at exactly zero.
With `gate == 0`, the gate gradient is proportional to the zero correction and
every head-weight gradient is proportional to the zero gate. Neither factor can
move on the first step, so both remain zero. The requested CNN or transformer
then behaves as a bare trunk even while aggregate training thresholds pass. A
Python value such as `1e-50` that underflows to float32 zero creates the same
absorbing state.

**Implementation boundary.** One active-model validator covers plain and
factored CNN and transformer heads. It preserves the shipped `0.1` recipe
exactly and refuses every value that converts to zero in the parameter dtype.
Structural presence is not enough: with the trunk frozen and a nonzero loss,
one backward step must give a finite nonzero update to at least one head
parameter or its gate.

**Code ownership.** The required
`emulator/experiment.py::validate_active_model_values` check owns the public
configuration refusal. `ResCNN` and `ResTRF` in `emulator/designs/plain.py`
and `TemplateResCNN` and `TemplateResTRF` in `emulator/designs/ia.py` own gate
storage, identity initialization, and defensive constructor checks.

**Acceptance evidence.** `gate_init` values `0`, `-0.0`, and a nonzero Python
value that underflows in float32 refuse before construction. The `0.1` recipe
remains exact. One head-only step moves ResCNN, ResTRF, and at least one
factored-template head away from the identity start. A control that bypasses
validation with a zero gate must show exactly zero head and gate gradients
while the trunk can still reduce loss. The registered Torch witness requires a
GPU-capable environment.

### ReLU is not valid after a zero-initialized head layer

**Rule.** An activation placed after a head layer whose output is initialized
to zero must satisfy both `a(0) == 0` and a finite, representably nonzero
`a'(0)`. The one active-model validator applies this rule to CNN and
transformer head pins in plain and template designs. ReLU remains valid in
trunks but is refused after a zero-initialized head layer.

**Reason.** Every structured head initializes its final mixing layer to zero
in `emulator/designs/plain.py::ResCNN.__init__`,
`emulator/designs/ia.py::TemplateResCNN.__init__`, and
`emulator/designs/blocks.py::TRFBlock.__init__`, then applies the selected
activation afterward. ReLU preserves the forward identity because
`a(0) == 0`, but Torch defines its derivative at zero as zero. The gradient
reaching the mixing weights is proportional to `a'(0)`, so the zeroed layer
never moves. ResCNN and TemplateResCNN become wholly inactive. In `TRFBlock`,
the attention output projection can move because no activation follows it,
while the MLP's zeroed final layer remains inactive. The model can therefore
improve while lacking the advertised MLP correction. H and tanh avoid this
failure because their origin derivatives are nonzero. ReLU's zero derivative
is intentional and cannot be repaired through numerical stabilization.

**Implementation boundary.** Validation refuses a ReLU head before model
construction and preserves exact identity at initialization without a random
perturbation. A separately named head-safe ReLU construction would require a
new scientific design. Power and gated-power activations qualify only when
their analytic origin derivative is implemented as one. Explanations in
the owning constructors must state this executed invariant. Schema refusal and
the one-step trainability witness are both required: the validator prevents
the invalid construction, and the behavioral witness detects a bypass or a
future activation regression.

**Code ownership.** The required
`emulator/experiment.py::validate_active_model_values` check owns the
configuration refusal. `emulator/activations.py::make_activation` owns
selection by activation name. The structured-head constructors in
`emulator/designs/plain.py`, `emulator/designs/ia.py`, and
`emulator/designs/blocks.py::TRFBlock` own zero initialization and defensive
compatibility checks.

**Acceptance evidence.** A registered Torch leg freezes or bypasses the trunk
and supplies a nonzero residual target. A ResCNN plus ReLU control that bypasses
validation must show exactly zero gradients and parameter changes for every CNN
head parameter and its gate. TemplateResCNN must reproduce the result. A
ResTRF plus ReLU control must show that `wo` moves while `mlp_lins[-1]` does
not. H and tanh controls retain exact identity at initialization and move the
zeroed layer after one step. Power and gated-power controls with the analytic
origin limit must also move. The schema accepts a ReLU trunk, refuses a ReLU
head before construction, and rejects any mutation that checks only
`a(0) == 0`.

## Padded heads preserve physical coordinate identity and inert padding

**Rule.** Plain and template padded heads preserve the original physical
coordinate of every kept value. Every padded position remains exactly inert
after every convolution, transformer block, and FiLM operation. The persisted
artifact stores both the coordinate map and the aligned validity mask.

**Reason.** Two mechanisms violate that rule when layout is reconstructed from
bin counts and invalid positions remain active:

1. **Rank can replace coordinate identity.** `pad_idx` constructed from
   `geometry.bin_sizes` alone places the `j`-th surviving entry of bin `g` at
   `g * max_bin + j` in `emulator/designs/plain.py::ResCNN.__init__` and
   `emulator/designs/ia.py::TemplateResCNN.__init__`. That rank is not
   necessarily the original angular slot. Two tomographic bins with the same
   kept count but different angular masks then receive identical layouts, and
   cross-bin channel mixing combines physically different angles in the same
   padded column. Counts cannot distinguish the two valid geometries, either
   during construction or after artifact rebuild.
2. **Padding can become active.** Padding starts at zero during the initial
   scatter, but convolution bias, activation, FiLM, and transformer updates
   act on the entire rectangle unless a validity mask is reapplied. In a
   two-block witness, cross-bin mixing can write a longer bin's value into an
   invalid column of a shorter bin. The next spatial kernel can move that value
   into a valid column, so the gathered correction depends on a nonexistent
   datum. Ragged single-bin segmentation through `n_tokens` exposes the same
   risk in its final partial token.

**Implementation boundary.** Each kept value scatters into its original
angular-coordinate slot, so equal-count bins with different masks remain
distinguishable. A boolean validity mask aligns with the padded tensor and is
reapplied after every CNN or transformer block and after FiLM. Attention and
MLP operations must not use invalid positions as keys, values, or latent
channels. Because angular positions form the feature dimension in this layout,
a conventional sequence-token attention mask alone is insufficient. Plain and
template CNN and transformer heads share the representation. The final partial
token created by ragged `n_tokens` segmentation is masked. A completely masked
physical bin remains an empty row instead of shifting every later bin to a new
coordinate. Masked LayerNorm uses only physical features; attention excludes a
query-key pair when its head has no shared physical feature. Equal-length input
without padding remains bitwise unchanged, as do rectangular CMB and grid
families. Before staging output files, save compares the model buffers with
the layout derived from the geometry and resolved recipe. A disagreement
cannot replace an existing valid pair. Rebuild performs the independent
comparison again. A structured-head artifact without the persisted fields is
refused rather than reconstructed from counts. Every explanation of zero
padding or matched angular scales must state the executed invariant.

**Code ownership.** Current scatter and gather behavior lives in the
constructors and `forward` methods of `ResCNN` and `ResTRF` in
`emulator/designs/plain.py` and `TemplateResCNN` and `TemplateResTRF` in
`emulator/designs/ia.py`. The required physical coordinate map and validity
mask belong to `emulator/geometries/output.py::build_shear_angle_map` and
`DataVectorGeometry.state` and `DataVectorGeometry.from_state`. Rectangular
CMB and grid families continue to define their layouts through each
geometry's `attach_head_coords` method. A new mask operation shared by the
four structured models must have one owner rather than four drifting copies.

<a id="padded-head-identity-layout"></a>

**Layout acceptance evidence.** The registered Torch-only
`padded-head-identity` gate requires equal-count bins with different masks to
produce different persisted maps. A one-block known-answer CNN mixes only
intended angular neighbors. A two-block routing witness returns exactly zero
when a value can travel only through an invalid slot, for both ResCNN and
TemplateResCNN. Multi-block ResTRF and TemplateResTRF keep invalid positions
exactly zero, and valid outputs are invariant to an injected invalid-slot
sentinel. Non-zero-preserving activations and FiLM shifts are exercised so a
test cannot pass merely because its chosen operation happens to preserve zero.
Additional witnesses cover a fully masked middle physical row, final
partial-token inertness, and an unchanged live rectangular CNN path.

<a id="padded-head-identity-artifact"></a>

**Artifact acceptance evidence.** The same gate uses the public save and
rebuild functions with a nonzero live CNN correction. It requires the reopened
prediction, geometry map, geometry mask, and model buffers to match exactly.
Saving a model with a missing mask or a geometry disagreement is refused
before any staging path is reserved, while a preceding valid pair remains
unchanged. A checkpoint that omits the fixed mask or disagrees with
the HDF5 geometry is refused before state loading. The workstation
`save-rebuild-drift` gate retains its real cosmic-shear structured-head round
trip and its refusal of an artifact written before map-and-mask persistence.
