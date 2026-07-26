# Models, designs, and scientific constraints

This note records the durable model rules that future changes must preserve.
It is a design specification, not a development diary. Every statement is
written as a durable rule, reason, code owner, or acceptance requirement.

## Words with a local meaning

**Artifact** = one saved emulator result: trained weights plus the facts needed
to rebuild them. **Identity** = the saved facts or byte fingerprints deciding
whether two datasets, runs, models, or coordinates are the same; never the
identity matrix. **Drift check** = a comparison of candidate behavior against a
declared reference, required to detect a deliberate change.

**Geometry** = the mapping between physical outputs and the coordinates a
network and loss use: axes, masks, scaling, fixed basis changes. To **whiten**
a residual is to transform it so its covariance is the identity; a **basis
transform** is the fixed matrix between two coordinate representations. A
**trunk** is the main network producing the initial prediction, a **correction
head** the smaller branch adding a structured residual to it. A **bin** is one
physically meaningful group of adjacent output coordinates.

A **token** is the feature vector for one bin or contiguous coordinate segment.
**Permutation equivariance** means reordering tokens reorders the outputs
instead of changing their values. FiLM identity initialization starts at
`gamma = 1`, `beta = 0`.

A **registry** maps accepted configuration names to the classes or functions
that own them. A **capability flag** is an explicit class property telling
shared code which geometry, data, or head behavior the class supports. `n(z)`
is a galaxy redshift distribution. CosmoLike produces the cosmic-shear data
vectors. The **board runner** executes registered gates and records their raw
results.

## How to use this note

Every independent technical contract answers four questions: what behavior is
required (**Rule**), what scientific or numerical failure the rule prevents
(**Reason**), which module owns the behavior (**Code ownership**), and which
exact checks separate the required behavior from a plausible but wrong
implementation (**Acceptance evidence**). One long section may answer the four
once and then list related subrules under bold labels. A new enforceable
behavior receives NO-GO when any of the four answers stays implicit.

The primary code owners are `emulator/designs/`, `emulator/losses/`, and
`emulator/activations.py`. Those files own how the code behaves; this note owns
the decisions behind it, the failures each rule prevents, and the designs that
measurement rejected. Gate evidence sits under stable HTML anchors so gate
metadata can address the exact contract.

## The architecture family

**Code ownership.** `emulator/designs/` and `emulator/losses/` own the model
family. Their module and class docstrings are the behavior specification, and
`emulator/designs/plain.py` carries the shape-flow diagrams for the trunk, the
`pad_idx` scatter and gather, the `W_fd` and `W_df` basis buffers, the `Conv1d`
stack, token segmentation, and the FiLM generator. Read them for how a model
works. This note records only the rules and reasons that no single module owns.
`emulator/activations.py` stays flat because code outside the family imports it
by path and drift checks patch it that way. Moves must preserve saved `.emul`
state dictionaries, saved class markers, and the explicit polynomial chaos
expansion (PCE) import.

**Rule.** Every model is selected through the shared registry and declares
structural capabilities for geometry, bins, parameters, and correction-head
type.

**Reason.** A parallel construction path or a class-name test can build a model
that trains but cannot be staged, saved, rebuilt, or served under the same
scientific coordinate identity.

**Acceptance evidence.** Configuration, initialization, phase, geometry,
save/rebuild, and family gates must exercise every declared capability. A
mutation that bypasses a capability or changes coordinate order must fail its
own named check.

**Registries and capabilities.** Intrinsic alignment (IA) describes the
alignment of galaxy shapes with the surrounding tidal field. The nonlinear
alignment model (NLA) and the tidal-alignment/tidal-torquing model (TATT) are
represented by the shared registry. Deployable TATT training additionally
requires a validated ten-template dataset; registry construction alone is not a
claim that such a production dataset is present. `MODELS` is keyed by
`(name, ia)`, and `IA_DESIGNS` owns each form's amplitude names, coefficient
function, and template count; a new IA form extends that table instead of
creating a parallel model path.

The capability flags `factored`, `needs_geom`, `needs_bins`, `needs_params`,
and `head_block` exist so that shared code never asks what class it holds. Each
flag is declared by the design and consumed by loss, training, staging, and
diagnostic callers, which is why the contract lives here and not in any one
module. `needs_params` in particular tells a diagnostic that a bare
`geometry.decode` would be wrong for that model.

**Structured heads are earned, not assumed.** A dense trunk cannot see a
permutation of the output coordinate, whereas shared head weights can use
neighboring-coordinate structure with fewer effective degrees of freedom. That
argument is not evidence: a coordinate-aware head, a FiLM option, or any other
head choice is supported only when a family-specific learning curve
demonstrates the benefit. One trained configuration establishes nothing.

Heads apply to cosmic shear, CMB, and both grid families. Scalar outputs stay
trunk-only because named scalar quantities do not share a coordinate axis.
Two-phase training belongs to every model with a correction head, not only to
an IA model; ResMLP is the only single-phase design. `training-stack.md` owns
the schedule options.

**Fixed buffers, not live geometry.** `W_fd` and `W_df` are model buffers
because a live geometry call inside `forward()` prevents stable CUDA graph
capture. A diagonal family — one whose chi-square metric acts independently on
each stored physical coordinate after per-coordinate scaling, without a dense
basis rotation — already whitens in physical order, so its basis change is the
identity and both buffers stay `None` rather than holding a square identity
matrix that would only waste memory.

**Identity at initialization.** The last layer of every correction branch is
initialized to zero, so the complete model equals the trunk at construction and
at the two-phase handoff. Preserving that identity requires `a(0) = 0`, and a
trainable branch also requires a finite, representably nonzero `a'(0)`. The
refusal contracts for the activations and the gate value that fail this appear
under *ReLU is not valid after a zero-initialized head layer* and *A zero
`gate_init` is an absorbing dead head* below.

**Head choices a docstring cannot justify alone.**

- Two stacked linear convolutions collapse to one linear map, so `act_mid`
  supplies the required intermediate nonlinearity.
- Convolution `groups` must respect probe-block boundaries. Separable
  convolution lowers parameter count but is slower on the measured path.
- A transformer token is a raw padded bin segment of natural width `max_bin`,
  26 for the Legacy Survey of Space and Time Year-1 (LSST-Y1) configuration,
  and `n_heads` must divide that width. The design carries no embedding or
  projection adapter, and the per-token MLP width equals the token width;
  reintroducing `mlp_width` requires a new design decision, and a shared MLP
  must preserve the intended permutation-equivariant behavior.
- A factored IA model separates exact amplitude columns from the emulated
  input, so FiLM must condition only on `x[:, :n_in]`. Conditioning on the
  appended amplitude columns would break the exact closed-form factorization.
- Cosmic-shear artifacts persist `bin_sizes` and `pm_kept`, the per-element
  flags that separate the xi-plus and xi-minus shear branches, in
  `DataVectorGeometry.state()`. `rebuild_emulator` refuses a structured-head
  artifact that predates those fields rather than reconstructing them.

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

`ai/gates/board.py` registers this gate's manifest, subprocess, five leg names,
and required capabilities; read its `Gate(id="head-activation-pin")` entry for
those. What is recorded here instead is the limit of what a passing result
proves:

- A manifest-bound GPU run is required. The four process/text legs establish
  option routing and refusal behavior only, and do not assert a parameter count
  even when the output contains related design information. Numerical
  prediction agreement needs a separate trained-artifact comparison; startup
  text cannot supply it.
- The driver follows the CosmoLike `.dataset` pointer to data-vector,
  covariance, mask, and n(z) siblings. Those are transitive reads outside the
  manifest hash.
- Golden selected-text equality compares selected log lines after their
  trailing wall-clock field is stripped, so it is not a raw-byte comparison. It
  requires a reviewed pinned base named in `board_config.json` and an assertion
  that both selected-line lists are nonempty, because the helper accepts two
  empty lists and equality of empty selections is not evidence.

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

`ai/gates/board.py` registers this gate's manifest, its two driver
configurations, the `ai/gates/checks/d5_training_behaviors.py` child, the seven
leg names, and the required capabilities. What is recorded here instead is the
limit of what a passing result proves:

- The two CPU legs establish factory-level numerical behavior on one fixed
  small regression: exact ReLU and Tanh values, identity initialization of the
  selected norms, finite strict loss descent, and a final loss below half the
  mean-only predictor's loss. They do not claim that the complete cosmic-shear
  jobs ran. The four process/text legs need Torch, CosmoLike, and a GPU, and
  remain workstation evidence that must not be inferred from the CPU result.
- The driver follows the CosmoLike `.dataset` pointer to data-vector,
  covariance, mask, and n(z) siblings, which are transitive reads outside the
  manifest hash.
- Golden selected-text equality compares selected log lines after their
  trailing wall-clock field is stripped, so it is not a raw-byte comparison,
  and it requires a reviewed pinned base and nonempty selected-line lists.

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
leave-one-out selection, fitted coefficients, and calibrated input domain, and
its docstrings define every `pce:` key, including a worked example of the
hyperbolic q-norm truncation that `p_max`, `r_max`, and `q` control.
`emulator/losses/pce.py` owns composition and target encoding.
`emulator/experiment.py` owns fitting and configuration validation.
`emulator/inference.py` owns rebuilt prediction.

**Acceptance evidence.** The `npce-training` gate and family identity gates
cover base fitting, supported forms, refusals, distinct sweep fits,
save/rebuild composition, and a mutation that omits the base contribution.

A neural model with a PCE base is called NPCE in the configuration and code.
The block stays outside `train_args` because `sweep_hyperparam` stages the base
once, so sweeping a PCE option there would change the option without refitting
the base. PCE is mutually exclusive with rescaling and factored IA.

A PCE base adds capacity rather than replacing the neural model, because the
shape modes are not low-degree polynomials. NPCE is supported infrastructure,
not an established way to lower the sample-efficiency floor. The fit keeps
degree low to limit Runge oscillation, retains only modes whose leave-one-out
(LOO) error clears `loo_max` (default `0.05`), stops early, and runs a greedy
residual-correlation search on the CPU using the closed-form
predicted-residual-sum-of-squares (PRESS) form of the LOO error. The function
keeps its public `select_lars_loo` name but does not execute the
least-angle-regression path algorithm.

Scalar, CMB, one-dimensional grid, and two-dimensional grid families wrap
`emulator/losses/pce.py::PCEResidualDiagChi2`, a subclass of `CmbDiagonalChi2`.
Their diagonal metric is the family chi-square, and a roughness term composes
with it because prediction minus target is the complete whitened residual.
`emulator/experiment.py::EmulatorExperiment._fit_diag_pce` owns fitting across
the four `build_geometry` branches. Every inference branch must use
`emulator/inference.py::_build_diag_decoder`, because a bare `geometry.decode`
would omit the saved base contribution.

Diagonal families support residual form only, because ratio form depends on a
dense covariance; `validate_pce(diagonal=True)` owns that refusal. CMB permits
NPCE only with `amplitude_law: none`, because the amplitude-law loss owns
target construction. A cosmic-shear conclusion does not transfer to the
matter-power family, whose PCE fits the nonlinear boost
`B(k, z) = P_nonlinear(k, z) / P_linear(k, z)` rather than either power
spectrum by itself.
<a id="npce-training-evidence"></a>

### Acceptance evidence: `npce-training`

**Rule.** The registered gate checks process results and selected NPCE text for
residual, ratio, refusal, and two-point-sweep configurations. Its smoke helpers
establish routing and refusal behavior only. Loss comparison and a distinct
base fit inside each sweep worker require separate executable witnesses.

`ai/gates/board.py` registers this gate's manifest, its residual, ratio,
refusal, and sweep driver runs, the nine leg names, and the required
capabilities. What is recorded here instead is the limit of what a passing
result proves:

- A manifest-bound GPU run is required, and the seven process/text legs
  establish routing and refusal behavior only. Numerical loss descent requires
  a numerical assertion; per-worker NPCE refitting requires worker-specific fit
  evidence; saved-artifact equivalence requires an executable
  rebuild-versus-base comparison. A printed rebuild instruction is process
  text, not a rebuild-versus-base comparison. None of these properties follows
  from process completion or selected startup text.
- The driver follows the CosmoLike `.dataset` pointer to data-vector,
  covariance, mask, and n(z) siblings, which are transitive reads outside the
  manifest hash. This gate does not read a saved NPCE artifact back.
- Golden selected-text equality requires a reviewed pinned base named in
  `board_config.json` and nonempty selected-line lists.

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
`loo_max`; equality does not pass. No-mode selection refuses the fit instead of
retaining a fallback. Support indices are unique, and selection stops when
every usable candidate is active.

Acceptance describes the base that will actually be saved: the saved float32
input bounds, training inputs converted to the serving float32 format,
coefficients rounded to float32 before scoring, and residual prediction through
the same dense float32 matrix multiplication used when serving. Once retained
modes are assembled, the complete multi-column coefficient matrix is scored
jointly; a joint failure removes the offending mode and the narrower matrix is
checked again. The fit refuses only if no mode remains.

**Reason.** Two specific wrong implementations stay finite and plausible. A
no-mode fallback in `PCEEmulator.from_training` could retain mode zero when no
mode satisfies `loo < loo_max`, so a persisted base would carry an error far
above the requested ceiling while the report claims the predicate held. In
`select_lars_loo`, an all-active candidate set can leave every score at `-1`,
where a second `argmax` would select column zero and append a duplicate support
index. Either behavior invalidates the claim that the base passed selection.

**Implementation boundary.** The no-mode fallback is forbidden, and a refusal
names the best attempted LOO, the threshold, and the modes tried. Every
recorded or retained LOO must be finite. Nothing may be manufactured to keep a
fit alive: no epsilon that turns a constant mode into evidence, no clipped
leverage that supplies a denominator, no duplicated support. `pce.loo_max`
needs an explicit finiteness check because a comparison with NaN does not
enforce positivity. `emulator/designs/pce.py` owns the remaining input,
shape, and termination conditions.

The PCE artifact state remains the six arrays `lo`, `hi`, `multi_index`, `C`,
`Vk`, and `Ybar`; it does not persist historical LOO scores. New fits are
certified before publication and their resolved PCE configuration records the
requested threshold. An older six-array artifact cannot be certified from those
arrays alone and should be retrained when its origin predates strict
saved-format acceptance.

**Code ownership.** `PCEEmulator.from_training` owns the retained-mode decision
and calls `select_lars_loo`, which owns support selection and its termination
rule. `emulator/experiment.py::validate_pce` owns the configuration value,
including finite `loo_max` validation.

**Acceptance evidence.** `ai/gates/checks/` and `ai/tests/` own the case lists.
The discriminating requirements are: a predictable control retains a real mode
with every LOO below threshold; a strict-threshold fixture refuses with `no
mode passed` and writes no artifact; NaN or infinity anywhere refuses; a
large-offset input witness proves that pre-round bounds cannot certify a
different saved design; large-coefficient witnesses prove that promoted float64
multiplication and separate per-mode products cannot hide float32 cancellation;
and a mode that fails only in the final joint matrix is removed while a
different passing mode remains. Production NPCE training requires the complete
set.

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
may still hold settings for an unused CNN or Transformer alternative; only the
selected architecture is checked.

**Reason.** `emulator/experiment.py::MODEL_BLOCK_KEYS` maps names but cannot
validate the mapped values before `build_specs` passes them to design
constructors. Malformed values then demote the requested architecture instead
of refusing:

- `model.trf.n_blocks: 0` builds an empty transformer-block list, so
  `TemplateResTRF.forward` leaves `t == t0` and the correction `t - t0` is
  zero. The trunk still trains and still passes aggregate collapse checks, so
  acceptance evidence must prove that a requested head cannot silently reduce
  to its trunk.
- A quoted `"false"` is a nonempty string and therefore truthy in Python.
  Without type validation, `rescale_kernel`, `separable`, `film`, or
  `shared_mlp` would be enabled by a value that reads as disabled.
- `model.cnn.n_blocks: 0` reaches `self.convs[-1]` and raises an unrelated
  `IndexError`; `model.trf.n_mlp_blocks: 0` reaches `self.mlp_lins[-1]` the
  same way.
- `n_heads: 0` makes `dim % n_heads` divide by zero. An incompatible positive
  value is guarded only by a Python assertion, which `python -O` removes, so
  public checks for `kernel_size`, `groups`, and geometry assumptions must
  raise typed exceptions instead.
- `float()` on `gate_init` would accept NaN or infinity. Because the correction
  starts at zero, `out = y + gate * corr` then produces NaN immediately.
- `int()` on `n_gates` or `n_tokens` would accept booleans, truncate fractions,
  accept numeric strings, and pass zero to code that allocates an empty gate
  tensor. Validation preserves the declared type instead of coercing.
- A zero-initialized correction layer sends zero into its activation, and
  `relu` has zero derivative there, so it cannot wake the requested head. `H`,
  `multigate`, `tanh`, `power`, and `gated_power` are live at that starting
  point. ReLU stays valid inside an MLP trunk.

**Implementation boundary.** One pure active-model value validator runs twice.
`EmulatorExperiment.from_config` calls it after selecting the model class and
before files, devices, sources, or construction. `build_specs` repeats it for
values produced by a parameter search; that second call has the output geometry
and can therefore check physical CNN grouping, Transformer token width, and
attention-head divisibility before translating values into constructor
arguments. Booleans must be YAML booleans, integral fields reject booleans,
strings, and fractions, and typed exceptions preserve every check under
optimized Python. `emulator/validation.py` and `emulator/experiment.py` own the
per-field conditions. An MLP trunk depth of zero stays valid because it
explicitly requests the documented linear-only trunk.

**Code ownership.** Shared exact-value helpers live in
`emulator/validation.py`. Model selection, the two validator calls, key
translation, and head-activation parsing live in `emulator/experiment.py`.
Constructors in `emulator/designs/plain.py`, `emulator/designs/ia.py`, and
`emulator/designs/blocks.py` repeat their local checks before allocating
learned layers, so a direct internal call cannot bypass the public path.

**Acceptance evidence.** `ai/tests/test_active_model_validation.py` owns the
case list and every error names the full dotted setting. The discriminating
requirements are: one public `from_config` example refuses before any file,
device, source, or model access; one `build_specs` example checks a searched
attention value against the built geometry before spec translation; direct
constructors run under ordinary and optimized Python; and small valid CNN and
Transformer examples prove that the requested blocks exist and that the final
zero-initialized correction layer receives a finite nonzero gradient and moves
on its first optimizer step.

### Transformer token width must be at least two

**Rule.** Every transformer token has width of at least two. The active-model
validator derives token widths from the physical geometry and refuses a
configuration whose maximum token width is below two.

**Reason.** `ResTRF.__init__` accepts `model.trf.n_tokens` from two through the
full output length, so `n_tokens == n_out` creates one scalar coordinate per
token and `max_bin == token_width == 1`. The divisibility check accepts
`n_heads: 1`, and `MODEL_BLOCK_KEYS` with `build_specs` exposes the
configuration.

At feature width one, LayerNorm is algebraically input-independent: the mean is
the scalar itself, the variance is zero, and every normalized value is zero
before the learned affine bias. Both pre-normalized branches in
`TRFBlock.forward` therefore discard the input. With `film: false`, every
attention and MLP branch output is a learned constant per token, independent of
cosmology; `TRFBlock(x) - x` is independent of `x` for any trained weights, and
stacking blocks preserves only an input-independent additive correction. Since
`ResTRF.forward` returns `t - t0`, the requested transformer can never learn a
sample-dependent correction while the ResMLP trunk still satisfies aggregate
collapse thresholds. That is a silent architecture demotion, and range and
divisibility checks cannot detect it.

**Implementation boundary.** The refusal names the output length, token count,
resolved width, and LayerNorm degeneracy. Plain and factored TRF constructors
share the invariant, and both must refuse before allocating any learnable
layer: a guard inside `TRFBlock` alone would fire only after `TemplateResTRF`
had already allocated its template trunk. `blocks.py::validate_trf_token_width`
owns the rule and its teaching error; `TRFBlock` repeats the check as defense
in depth for direct construction. Adjacent accepted configurations stay
unchanged, and no padding, embedding, or projection silently substitutes a
different design.

**Acceptance evidence.** A registered Torch leg on a GPU-capable environment
uses a single-bin `N=4, n_tokens=4, n_heads=1, film=false` configuration and
requires refusal before model construction. A control that bypasses validation
must produce identical corrections for two distinct `t0` rows and a zero
correction Jacobian; the adjacent `n_tokens=3` configuration must construct and
produce an input-dependent correction; plain and factored paths must agree; and
a mutation that restores only the range and divisibility checks must construct
but fail the behavioral witness. The CPU companion
`ai/tests/test_trf_token_width.py` repeats early refusal under ordinary and
optimized Python, and a mutation that removes only the factored model-level
call must allocate trunk layers before the block-level guard raises. The
`cmb_identity.py` check keeps its width-20 ResTRF save and rebuild leg; local
CPU checks do not replace the registered integration leg.
Local CPU checks do not replace the registered integration leg.

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
component has derivative one at the origin, including the `p=1` identity case.

**Reason.** The sign-based expression
`torch.sign(x) * ((1.0 + abs(x)) ** p - 1.0) / p` has the right value and the
wrong Jacobian at `x=0`. Both `sign(x)` and `abs(x)` have zero derivative
there, and the inner magnitude is zero, so automatic differentiation returns a
zero power-component derivative. Identity-initialized correction layers,
padding, and new channels deliberately create exact zeros, and a forward
identity check cannot detect this gradient-absorbing point.

**Implementation boundary.** Express the power component as `x` times an even
magnitude ratio, `psi_p(x) = x * ((1 + abs(x))**p - 1) / (p * abs(x))`, whose
analytic limit at zero is one. Compute the near-zero ratio with `log1p` and
`expm1` or a justified series; an unguarded `0/0` branch is forbidden.
Constructors require finite positive `p_min < p_max` so a malformed direct call
cannot drive the denominator toward zero. Forward values away from the
near-zero neighborhood stay unchanged. Documentation may state the derivative
only after executable evidence proves it.

**Code ownership.** `emulator/activations.py::PowerGatedActivation` and
`GatedPowerActivation` own the forward formulas and trainable power bounds, and
`make_activation` owns selection by configuration name. Head-compatibility
validation stays with the active-model validator.

**Acceptance evidence.** A registered Torch leg on a GPU-capable environment
compares H, power, and gated-power values and input gradients at
`x = [-epsilon, 0, +epsilon]` under default initialization. With `p=1`, the
power component equals `x` and has derivative one, including at exactly zero; a
zero preactivation inside a small residual block transmits a nonzero gradient;
float64 `gradcheck` covers several learned powers; and a mutation that restores
`sign(x) * f(abs(x))` must fail specifically at the zero-Jacobian assertion.

## NPCE refuses queries outside its calibrated domain

**Rule.** A PCE base accepts a query only inside its persisted calibration
domain, apart from an explicitly defined floating-point tolerance. Training,
validation, and rebuilt-artifact inference apply the same policy.

**Reason.** `PCEEmulator.forward` maps whitened inputs into a fitted Legendre
box and clamps every mapped coordinate to `[-1, 1]`. An input one rounding unit
beyond the boundary and an arbitrarily distant cosmology therefore map to the
same boundary point, and the output stays finite even though the base evaluated
a different cosmology. A neural residual refiner sees the original input, but
its training target already includes the base's saturation, so no rule
guarantees correction of an arbitrarily clipped base. The persisted `lo` and
`hi` already carry the calibration data needed to refuse instead.

**Implementation boundary.** Persist a named PCE domain policy. Scientific
serving refuses points outside the calibrated whitened box, with only a
documented scale-aware tolerance for roundoff; optional clipping inside that
tolerance never substitutes for validation. A refusal names the stored
parameter coordinate, whitened value, allowed `[lo, hi]` interval, and
overshoot, mapped to the input-geometry record when available. Persisted bounds
must be finite, one dimensional, aligned with the PCE input width, and satisfy
strict `lo < hi`. The resolved fit and evaluation record exact boundary hits
and near-tolerance points. Leave-one-out selection judges polynomial quality
inside the domain; the domain policy decides whether a query belongs there at
all.

**Code ownership.** `PCEEmulator.from_training` owns the calibrated bounds and
saved policy, and `PCEEmulator.forward` owns the check before basis evaluation.
`emulator/results.py::rebuild_emulator` and `emulator/inference.py` must
preserve and apply the same saved policy after rebuild.

**Acceptance evidence.** Two far-out inputs on the same side of a boundary must
both refuse rather than collide. `ai/gates/checks/` owns the remaining cases —
values below `lo` and above `hi` in each dimension, nonfinite bounds, equal
bounds, shape mismatch, exact endpoints, and a one-unit-in-the-last-place
tolerance control. Training and rebuilt-artifact inference must agree, and the
witness covers residual NPCE on one diagonal family and one dense-covariance
family.

### A zero `gate_init` is an absorbing dead head

**Rule.** `gate_init` must be finite, real, non-boolean, and representably
nonzero after conversion to the parameter dtype. Positive sign is not required;
a negative gate is equivalent up to the correction sign.

**Reason.** A structured head returns `trunk + gate * correction` and its
correction branch starts at exactly zero. With `gate == 0`, the gate gradient
is proportional to the zero correction and every head-weight gradient is
proportional to the zero gate, so neither factor can move on the first step and
both stay zero forever. The requested head then behaves as a bare trunk while
aggregate training thresholds still pass. A Python value such as `1e-50` that
underflows to float32 zero creates the same absorbing state.

**Implementation boundary.** One active-model validator covers plain and
factored CNN and transformer heads. It preserves the shipped `0.1` recipe
exactly and refuses every value that converts to zero in the parameter dtype.
Structural presence is not enough: with the trunk frozen and a nonzero loss,
one backward step must give a finite nonzero update to at least one head
parameter or its gate.

**Code ownership.** `emulator/experiment.py::validate_active_model_values` owns
the public configuration refusal. `ResCNN` and `ResTRF` in
`emulator/designs/plain.py` and `TemplateResCNN` and `TemplateResTRF` in
`emulator/designs/ia.py` own gate storage, identity initialization, and
defensive constructor checks.

**Acceptance evidence.** `gate_init` values `0`, `-0.0`, and a nonzero Python
value that underflows in float32 refuse before construction, and the `0.1`
recipe stays exact. One head-only step moves ResCNN, ResTRF, and at least one
factored-template head away from the identity start. A control that bypasses
validation with a zero gate must show exactly zero head and gate gradients
while the trunk can still reduce loss. The registered Torch witness requires a
GPU-capable environment.

### ReLU is not valid after a zero-initialized head layer

**Rule.** An activation placed after a head layer whose output is initialized
to zero must satisfy both `a(0) == 0` and a finite, representably nonzero
`a'(0)`. The one active-model validator applies this to CNN and transformer
head pins in plain and template designs. ReLU stays valid in trunks and is
refused after a zero-initialized head layer.

**Reason.** Every structured head initializes its final mixing layer to zero
and then applies the selected activation. ReLU preserves the forward identity
because `a(0) == 0`, but Torch defines its derivative at zero as zero, and the
gradient reaching the mixing weights is proportional to `a'(0)`, so the zeroed
layer never moves. ResCNN and TemplateResCNN become wholly inactive. In
`TRFBlock` the attention output projection can still move, because no
activation follows it, while the MLP's zeroed final layer stays inactive — so
the model improves while lacking the advertised MLP correction. H and tanh
avoid this because their origin derivatives are nonzero. ReLU's zero derivative
is intentional and cannot be repaired through numerical stabilization.

**Implementation boundary.** Validation refuses a ReLU head before model
construction and preserves exact identity at initialization without a random
perturbation. A separately named head-safe ReLU construction would require a
new scientific design. Power and gated-power activations qualify only when
their analytic origin derivative is implemented as one. Explanations in the
owning constructors must state this executed invariant. Both the schema refusal
and the one-step trainability witness are required: the validator prevents the
invalid construction, and the witness detects a bypass or a future activation
regression.

**Code ownership.** `emulator/experiment.py::validate_active_model_values` owns
the configuration refusal and `emulator/activations.py::make_activation` owns
selection by name. The structured-head constructors in
`emulator/designs/plain.py`, `emulator/designs/ia.py`, and
`emulator/designs/blocks.py::TRFBlock` own zero initialization and defensive
compatibility checks.

**Acceptance evidence.** A registered Torch leg freezes or bypasses the trunk
and supplies a nonzero residual target. A ResCNN plus ReLU control that
bypasses validation must show exactly zero gradients and parameter changes for
every CNN head parameter and its gate, and TemplateResCNN must reproduce that;
a ResTRF plus ReLU control must show that `wo` moves while `mlp_lins[-1]` does
not; H, tanh, power, and gated-power controls retain exact identity at
initialization and move the zeroed layer after one step. The schema accepts a
ReLU trunk, refuses a ReLU head before construction, and rejects any mutation
that checks only `a(0) == 0`.

## Padded heads preserve physical coordinate identity and inert padding

**Rule.** Plain and template padded heads preserve the original physical
coordinate of every kept value. Every padded position stays exactly inert after
every convolution, transformer block, and FiLM operation. The persisted
artifact stores both the coordinate map and the aligned validity mask.

**Reason.** Two mechanisms violate that rule when layout is reconstructed from
bin counts and invalid positions stay active:

1. **Rank can replace coordinate identity.** A `pad_idx` built from
   `geometry.bin_sizes` alone places the `j`-th surviving entry of bin `g` at
   `g * max_bin + j`, and that rank is not necessarily the original angular
   slot. Two tomographic bins with the same kept count but different angular
   masks then receive identical layouts, so cross-bin channel mixing combines
   physically different angles in the same padded column. Counts cannot
   distinguish the two valid geometries, either during construction or after
   rebuild.
2. **Padding can become active.** Padding starts at zero during the initial
   scatter, but convolution bias, activation, FiLM, and transformer updates act
   on the entire rectangle unless a validity mask is reapplied. In a two-block
   witness, cross-bin mixing can write a longer bin's value into an invalid
   column of a shorter bin, and the next spatial kernel can move that value
   into a valid column — so the gathered correction depends on a nonexistent
   datum. Ragged single-bin segmentation through `n_tokens` exposes the same
   risk in its final partial token.

**Implementation boundary.** Each kept value scatters into its original
angular-coordinate slot, so equal-count bins with different masks stay
distinguishable. A boolean validity mask aligns with the padded tensor and is
reapplied after every CNN or transformer block and after FiLM. Attention and
MLP operations must not use invalid positions as keys, values, or latent
channels; because angular positions form the feature dimension in this layout,
a conventional sequence-token attention mask alone is insufficient. Masked
LayerNorm uses only physical features, and attention excludes a query-key pair
when its head has no shared physical feature. Plain and template CNN and
transformer heads share the representation, the final partial token created by
ragged `n_tokens` segmentation is masked, and a completely masked physical bin
stays an empty row instead of shifting every later bin to a new coordinate.
Equal-length input without padding stays bitwise unchanged, as do rectangular
CMB and grid families. Save compares the model buffers with the layout derived
from the geometry and resolved recipe before staging output files, rebuild
repeats that comparison independently, and a disagreement cannot replace an
existing valid pair. A structured-head artifact without the persisted fields is
refused rather than reconstructed from counts. Every explanation of zero
padding or matched angular scales must state the executed invariant.

**Code ownership.** Scatter and gather behavior lives in the constructors and
`forward` methods of `ResCNN` and `ResTRF` in `emulator/designs/plain.py` and
`TemplateResCNN` and `TemplateResTRF` in `emulator/designs/ia.py`. The physical
coordinate map and validity mask belong to
`emulator/geometries/output.py::build_shear_angle_map` and to
`DataVectorGeometry.state` and `DataVectorGeometry.from_state`. Rectangular CMB
and grid families keep defining their layouts through each geometry's
`attach_head_coords` method. A mask operation shared by the four structured
models must have one owner rather than four drifting copies.
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
