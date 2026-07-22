# The training stack: losses, schedules, phases, moving averages, sizing, and evidence

This note defines the mandatory training contracts. It records stable behavior,
scientific invariants, and acceptance checks. It is not a diary. A training
change receives **GO** only when every affected contract has executable
evidence and a deliberate mutation proves that the evidence can fail.

## Vocabulary used throughout this note

An **exponential moving average (EMA)** is a smoothed copy of model weights
that gives more weight to recent updates. The **reverse Huber (BerHu) loss** is
the piecewise loss defined below. A **polynomial chaos expansion (PCE)** is an
analytic polynomial base model; a **neural polynomial chaos expansion (NPCE)**
is a neural correction to that base. The **cosmic microwave background (CMB)**
is relic radiation whose spectra form one output family. **Intrinsic alignment
(IA)** is the systematic alignment of galaxy shapes and forms a nuisance-model
family in the loss functions.

A **command-line interface (CLI)** is the set of terminal options accepted by
a driver. YAML is the human-readable settings-file format used by those
drivers. A **graphics processing unit (GPU)** is an accelerator used for
training. **Random-access memory (RAM)** is the computer's working memory.
CUDA is NVIDIA's GPU computing platform. The Apple Metal Performance Shaders
accelerator backend is named in full here because MPS elsewhere in this
library means matter-power spectrum.

CoCoA is the surrounding cosmological-analysis installation. CosmoLike is the
likelihood calculation inside that installation that produces and evaluates
several emulator data-vector families.

**Automatic mixed precision (AMP)** lets PyTorch use lower-precision numbers
for selected operations. **Autocast** chooses those operation-specific number
formats, and a **gradient scaler** rescales a float16 loss to protect small
gradients. **Eager execution** runs each PyTorch operation immediately;
**compiled execution** prepares an operation graph for reuse. `float16` and
`bfloat16` are two 16-bit floating-point formats. **Fused execution** combines
several optimizer operations into one accelerator kernel.

A **study** tests a collection of related configurations. A **point**, also
called a **trial**, is one configuration and its run. A **sweep** is a
systematic set of those points. A **worker** is one process that executes one
study point. A **lane** is one concurrent worker slot with its own resident
model and buffers. **Permanent
per-lane state** means memory that stays allocated for the lifetime of that
slot, rather than memory used by one batch. A **capacity token** reserves one
lane's share of a limited computing resource. A **manifest** is the complete
structured record of facts that identify a study. A **digest** is a fixed-size
fingerprint calculated from exact bytes. A **journal** is the on-disk study
record used for restart. A **progress watchdog** is a timer that treats a live
but non-advancing worker as stuck. A **pristine source state** is an unchanged
copy captured immediately after loading a source artifact.

A **probability density function (PDF)** describes how probability is
distributed over values. An **abstract syntax tree (AST)** is Python's parsed
structural representation of source code. A **gate** is a named validation job
whose required result is written before it starts. A **fixture** is the fixed
input setup used by a gate. A **control** is a valid case that must pass. A
**mutation** deliberately restores one forbidden behavior and must fail. A
**smoke command** is a short public run that proves startup and routing, not
numerical correctness. `UNAVAILABLE` means that a named evidence action did
not run or could not prove its claim. A **schema** is the required fields,
types, and meanings of a structured record. An **identity** is the set of facts
used to decide whether two runs, studies, or products are the same.

In gate evidence, a **reference comparison** compares a new run with a declared
reference run. An **exact-output-line comparison** compares the specific,
nonempty command-output lines named by the gate. It removes a timing field only
when that removal is stated explicitly.

## Ownership map

- `emulator/losses/` owns score construction and loss transforms.
- `emulator/training.py` owns validation reduction, optimizer updates,
  schedules, phase execution, selection, and restoration.
- `emulator/experiment.py` owns configuration resolution and construction.
- `emulator/batching.py` owns placement and streaming arithmetic.
- `emulator/diagnostics.py` owns diagnostic calculations.
- Public family drivers own command-line policy and product publication.
- `ai/gates/checks/` owns independent acceptance evidence.

One rule has one implementation owner. Drivers do not re-declare a capability
that belongs to a loss, diagnostic, geometry, or experiment object.

## Loss family

The accepted loss modes are `sqrt`, `chi2`, `sqrt_dchi2` (the square root of a
delta-chi-squared score), `berhu`, and `berhu_capped`. Loss transforms apply
during training only. Validation, selection, EMA evaluation, and diagnostic
scoring use raw per-sample
chi-squared values.

For a chi-squared value `c` and knot `k`:

```text
sqrt:          sqrt(c)
BerHu:         sqrt(c)                         when c <= k
               (c + k) / (2 sqrt(k))          when c > k
```

For cap `K`, `berhu_capped` keeps the BerHu definition through `K` and uses

```text
(2 sqrt(K c) + k - K) / (2 sqrt(k))
```

above `K`. The transform is continuously differentiable at both joins. The
knot is measured in delta-chi-squared units; a literature BerHu delta equals
`sqrt(k)`. The transform applies to one sample's total misfit, not element by
element.

Configuration lives under `train_args.loss`:

- `mode` selects the loss;
- `berhu` accepts `knot`, `cap`, and optional `anneal`;
- `roughness` accepts `lam` and `period_cut` and is valid only for the CMB
  family.

An absent loss block means `sqrt`. A plain `berhu` mode may accept an unused
cap so one block can serve a mode sweep. A block named for the active
`berhu_capped` mode may be canonicalized to the shared `berhu` block on a
copy. Any incompatible duplicate or mismatch raises with a paste-ready repair.

### Exact-fit square-root behavior

Every square-root expression that can receive zero uses the shared safe
square-root implementation. The forward value remains exactly `sqrt(c)` for
valid input, including `sqrt(0) == 0`. The gradient at exact zero is finite and
exactly zero. This includes both BerHu lower branches, the anneal arm, and the
masked capped-region expression. Positive values retain their ordinary
analytic derivatives.

### Chi-squared domain

Training, validation, source scoring, and every public diagnostic use one
score-domain rule:

```text
band = max(1.0e-6, 32 * eps(compute_dtype) * kept_width)
```

`kept_width` counts the kept coordinates one row sums over in the chi-square
contraction — the kept data-vector length for every CosmoLike-style loss —
so the band grows linearly with that count, not with its square. The dtype is the dtype in which the contraction was computed,
not a later storage or reporting upcast.

- A finite score at or above zero is unchanged.
- A finite negative score inside the band becomes exact zero.
- A finite score below the band, NaN, or either infinity is invalid.
- Training converts invalid producer output into a non-finite loss so the
  per-step finite guard refuses it without leaving compiled code.
- Evaluation and diagnostics raise before any mean, median, threshold,
  ranking, plot, or publication operation. The error names the boundary,
  producer, bad count, first row positions, minimum value, and band.

The band may be widened only after a measured valid control and a recorded
forward-error derivation. Convenience is not a reason to weaken it.

### Physical contractions and geometry precision

Network output and packed targets remain in the model compute dtype, normally
float32. Immediately before a physical Mahalanobis contraction—the
covariance-weighted squared residual—the residual is cast to the exact dtype
and device of the precision tensor. One shared helper owns the cast and
contraction for PCE ratio, transfer sum or gain,
plain transfer, factored transfer, and future physical-space losses. A
float64 precision tensor produces float64 chi-squared output; the float32
path remains unchanged.

Structured heads cast geometry basis-transform buffers into the trunk-output
dtype at the head boundary. Geometry precision remains unchanged for the
loss. Forward and inverse basis transforms require independent numerical
checks with declared expected values. Residual convolutional (ResCNN),
residual transformer (ResTRF), and their template variants TemplateResCNN and
TemplateResTRF must complete forward and backward with a float64 output
geometry.

### Roughness is a CMB capability

Roughness filtering assumes an ordered multipole axis. Method presence or
inheritance does not establish that scientific meaning. Only `data.cmb` may
use `train_args.loss.roughness`. Scalar, grid, grid2d, and ordinary CosmoLike
runs reject the block before staging, PCE fitting, model construction, or
training. CMB with NPCE and no amplitude law keeps the same penalty.

## Shared schedules

One validator and one evaluator serve trim, focus, BerHu blending, and EMA
horizon schedules. A schedule block is active by presence, not by a separate
Boolean. Each phase restarts its schedule at that phase's first epoch.

Shared keys are `hold_epochs`, `anneal_epochs`, and `shape`.
`hold_epochs` is a nonnegative integer, `anneal_epochs` is a positive integer,
and Boolean values are rejected. Supported shapes are `const`, `linear`,
`cosine`, and `step`.

Trim adds finite `start` and `end` values in `[0, 1)`. Focus adds finite
nonnegative `start` and `end` plus finite positive `kappa`. Trim and focus may
use `const`. BerHu and EMA are internally fixed zero-to-one schedules, so
`const` would disable the feature and is rejected. Omitting their anneal block
means the full feature is active from its normal start.

The step evaluator is direction-aware. A decreasing schedule preserves the
existing 0.01-grid sequence. An increasing schedule advances through strict
interior values and reaches the endpoint only at the final scheduled epoch.
Unknown shapes raise defensively even after configuration validation.

## Phase resolution

The `trunk` block configures the shared feature extractor, and the `head` block
configures the output-specific stage. These blocks accept exactly `lr`,
`scheduler`, `loss`, `trim`, `focus`, `clip`, `rewind`, and `ema`.

- `lr` is an overlay: specified phase fields replace top-level values, while
  unspecified fields inherit. A phase cannot redefine the run-global
  `bs_base` batch anchor.
- Every other phase block is a complete replacement for that setting.
- A phase scheduler replaces scheduler keyword arguments but keeps the
  scheduler class; `cls` is rejected inside a phase.
- An absent phase `ema` inherits. `ema: null` explicitly disables inheritance.

Two-phase capability is determined from the class capability, not from a
model-name string. Trunk-and-head designs include plain ResCNN and ResTRF on
every supported family and the factored-IA templates. The residual multilayer
perceptron (ResMLP) and its IA variants are single phase.

Single-phase demotion occurs in one pure resolver before training. It works on
a copy, removes `head`, `trunk_epochs`, and `freeze_trunk`, and promotes
`trunk.X` to top-level `X`. The head block is validated before it is removed.
A sweep over `head.*`, `trunk_epochs`, or `freeze_trunk` is refused for a
single-phase model and points to the promoted key.

`freeze_trunk` defaults to true. False means the second phase trains trunk and
head jointly; the phase role remains head while the model state is joint. A
per-head activation pin requires a positive trunk phase and a frozen trunk.

The top-level PCE block fits a closed-form base on every legal family. CMB
requires no amplitude law. The named model remains the neural correction
model, including its heads, phases, and full loss surface.

Transfer refinement permits one phase. Configuration that requests a
trunk-and-head transfer refinement is refused. A future two-phase transfer
design must establish complete correction-model trainability before optimizer
construction and must report trainable base, trunk, and head counts
separately.

## Resolved run record

Every display and product uses the resolved configuration that execution
consumes. A class describes its own resolved specification. A constructible
architecture missing its description fails during import or construction.

The immutable resolved record contains at least:

- family, model, activation, optional head activation, and gate counts;
- loss mode, thresholds, roughness, and schedules;
- phase topology and effective epoch/update counts;
- optimizer class, resolved fused state, scheduler protocol, and learning
  rates;
- device, autocast dtype, and gradient-scaler policy;
- EMA horizon and schedule;
- PCE, fine-tune, or transfer composition and source identity;
- effective training rows and dropped global tail; and
- study or sweep identity.

Console status lines, artifact metadata, table headers, figure labels, worker
arguments, and tuning reports read this record. Raw optional CLI fields never
masquerade as executed state. An activation sweep records `swept` plus ordered
values; a fixed run records the activation that actually ran.

## Selection record

The training loop returns one explicit selection record alongside plotting
histories. It contains pass identity, selected epoch or phase baseline,
threshold vector and selected index, fractions, mean, median, and whether the
selected weights are raw or EMA weights. Drivers, artifacts, console output,
and tuning objectives consume this record and never recompute a winner from
histories.

Epoch-zero and phase baselines remain outside trained-epoch histories. A
restored baseline is labeled as a baseline, not disguised as a trained epoch.
The record always describes the exact weights that ship.

Thresholds normalize once to a nonempty one-dimensional finite real tensor.
Boolean, repeated, unordered, empty, or wrong-rank input is rejected under the
documented ordering policy. Reports name the actual selected threshold; a
hard-coded `0.2` field is permitted only when the selected threshold is
exactly `0.2`. Diagnostics that intentionally use the scientific 0.2 target
remain separate from training selection.

The median is the ordinary 50th percentile. For even sample counts it is the
mean of the two central values; for odd counts it is the central value. One
helper supplies validation, scheduler input, tie-breaking, histories, and
gate references. Published means and medians are reduced in float64 from the
accepted row tensor. Every published mean, median, and fraction is checked for
finiteness after reduction.

## EMA and rewind

`train_args.ema` contains `horizon_epochs` and optional `anneal`. The horizon
is measured in epochs. Per-step beta, the fraction retained from the preceding
averaged weights, is derived from the horizon and actual `steps_per_epoch`.
This makes beta independent of batch size. An absent EMA block leaves the
non-EMA loop behavior unchanged.

The best snapshot is one coupled object containing model parameters,
optimizer state, and averaged parameters. Rewind restores all three. An
operation that makes a step no longer part of the live trajectory must also
remove that step from the average.

Raw-model validation drives `ReduceLROnPlateau`. EMA validation drives
selection and reported metrics. Evaluation swaps weights in place; it never
rebinds `.data`, because compiled execution and CUDA graphs keep references to
the original tensor storage. The shipped model contains the selected EMA
weights when EMA is enabled.

EMA annealing scales the horizon. While `scaled_horizon * steps_per_epoch < 1`,
beta is zero and the average tracks the live model exactly.

The canonical accepted update order is:

1. scale the loss when float16 scaling is active;
2. backward;
3. unscale gradients;
4. reject non-finite unscaled gradients;
5. clip unscaled gradients when requested;
6. perform the optimizer step;
7. apply the fine-tune anchor; and
8. update EMA from the post-anchor weights.

If the gradient scaler skips an optimizer step, neither the anchor nor EMA
advances. With beta zero and a nonzero anchor, EMA must equal the analytically
anchored value.

## Compiled-scalar discipline

Every scalar consumed by compiled loss code is a zero-dimensional device
tensor created per pass or filled in place per epoch. This includes kappa,
knot, cap, trim, focus, and schedule blend values. A captured Python float can
silently prevent reuse of the compiled graph.

EMA beta is the deliberate exception. It is an eager per-epoch Python float
because its interpolation runs outside the compiled graph. This exception is
not a template for compiled schedules.

## Optimizer and scheduler protocol

Optimizer numeric controls are validated before construction: learning rate
is finite and positive; weight decay is finite and nonnegative; epsilon is
finite and positive; betas are finite, non-Boolean, and inside the documented
range. A malformed value cannot reach an optimizer.

Weight decay is module-role based. Decay applies exactly to `.weight` on
`Linear`, `Conv1d`, and `BinLinear`. Biases, affine gains, and activation
parameters remain undecayed. Unlisted future modules default to no decay.

The supported optimizer surface is closure-free Adam/AdamW-style stepping.
Here a closure would ask the optimizer to evaluate the model again during one
step. An explicit `fused` value is preserved when legal. Automatic fused mode
is injected only for a class whose signature and backend support it. A
closure-required optimizer is refused before construction. Ordinary and
transfer-refinement factories use the same capability decision.

Each scheduler has a persisted protocol:

- cadence is `per_update` or `per_epoch`;
- metric requirement and metric source are explicit; and
- each pass records its resolved update horizon.

A per-update scheduler advances only after an accepted optimizer update. A
skipped step or empty chunk does not advance it. A per-epoch scheduler advances
once after a complete accepted epoch. Plateau scheduling receives the shared
ordinary median from raw-model validation. Warmup has one owner and cannot be
advanced by both the loop and scheduler.

`float16` AMP on the Apple Metal Performance Shaders accelerator backend
requires a supported gradient-scaling path. If the pinned Torch version cannot
scale correctly on that backend, `use_amp: true` is refused before training
with an actionable message. CUDA or CPU `bfloat16` does not enter the
float16-only scaling branch. Full-precision and bfloat16 paths retain their
existing numerical behavior.

## Data counts and batch sizing

`data.n_train` and `data.n_val` are required absolute row counts applied after
parameter cuts. Positive counts are enforced. Under one split seed, a smaller
selection is a prefix of a larger selection on the sorted in-memory path.

Batch size and epoch count are strict positive integers. Boolean and
fractional values are rejected. `bs > n_train` is refused before model or
loader setup.

Validation batch size is derived, not configured:

```text
k = ceil(n_val / 1024)
eval_bs = ceil(n_val / k)
```

The result is capped by the validation loader's own safe chunk. The final
validation batch is padded only to preserve one compiled shape; all real rows
are scored exactly once and metrics are partition-invariant.

Train and validation chunks are sized independently. Every validation request
uses `data["val"]["load"]`, never the training chunk. A test forces different
train and validation placement regimes and proves that no request exceeds the
validation limit.

Memory placement may alter input/output (I/O) grouping but not epoch semantics.
Every non-final chunk contains a whole number of training batches. At most one
global tail of `n_train % bs` rows is omitted. Every regime executes
`n_train // bs` steps and records that count. Padding duplicate training rows
into the objective is forbidden.

## Memory accounting

Sizing is observational. A sizing call preserves parameters, registered
buffers, every module's train/eval flag, and CPU/device random state exactly.
Dummy input uses the model's declared dtype and device. Byte totals are
computed per tensor as `numel * element_size`; mixed dtypes and registered
buffers are counted correctly.

The planner separates:

- permanent per-lane state;
- resident model, geometry, loss, precision, and PCE state;
- input batch bytes;
- model-output bytes;
- actual packed-target bytes;
- temporary loss working memory; and
- projected gradient and optimizer state.

Packed-target width comes from the same loss object that stages targets. One
batch costs `input + model output + actual target + temporary loss memory`,
not two copies of model-output width. If resident state plus one complete batch
exceeds the allowance, refuse and report required bytes, available bytes, and
every term.
Never force a batch count of one when one batch does not fit.

Sequential allocators receive the budget remaining after earlier resident
allocations. Resource estimates use actual output width, target width,
geometry/loss buffers, and dtype. Dense-CosmoLike and wide-diagonal families
require separate numerical checks with declared expected values against
measured peak allocation.

Capacity tokens are acquired before any job-sized GPU allocation. Either
token-scoped staging owns all job-sized setup or tokens cover the complete
lifetime of a lane-local staged experiment. Permanent per-lane state is
multiplied by the actual lane count before capacity is advertised. Failure at
setup, acquisition, or execution releases exactly the acquired resources and
reaps sibling processes.

## Complete state for repeated training

Every training invocation establishes the full loss-object state. Block
absence means clear the feature; it never means inherit state from a previous
point. This rule applies to roughness and every conditional `configure_*`
method, including law and rescaling state.

A repeated transfer-refinement study captures one pristine source state `W0`,
including parameters and buffers, immediately after artifact load. Each
point restores that state in place before loader construction or geometry
rebuild, resets live mode to false, restores evaluation mode and original
`requires_grad` flags, clears gradients, and resets the recorded source
snapshot. A failed point follows the same reset discipline. Point order,
worker count, lane assignment, and prior failures cannot change a later
point's starting state.

The pristine digest is part of study and sweep identity. Every point's anchor
reference must match it. Runs without transfer refinement retain their
existing behavior.

## Configuration totality

One pure validator serves ordinary training, tuning, and sweeps. It enforces
exact top-level and nested key sets before device selection, staging, source
loading, or artifact writes. The accepted root-key list comes from a complete
search of shipped YAML and deliberately supported CoCoA blocks that the driver
forwards unchanged. Unknown keys raise with the misspelling and close matches
from `difflib.get_close_matches`.

Run-control leaves use strict types and finite-domain checks. Phase overrides
route through the same leaf validators. A quoted Boolean is not a Boolean.
NaN and infinity are rejected before ordered comparisons. Validated values,
not raw input, enter the resolved record.

Range leaves have exactly `[default, minimum, maximum, kind]`, where `kind` is
`int`, `float`, or `log`. Values are finite, non-Boolean numbers;
`minimum < maximum`; the default lies inside the interval; integer ranges use
integral values without truncation; and log ranges have a positive minimum.
The same normalized range feeds ordinary defaults, search defaults, and
suggestions. Errors name the dot-separated path to the setting.

An alias that conflicts with the current settings structure is refused with a
paste-ready YAML block containing
the offending value and current replacement. Use `ValueError`, not `KeyError`,
so multiline guidance remains readable. A natural, self-consistent spelling
may be canonicalized on a copy when the accepted semantics are unambiguous.

`make_chi2` validates `rescale` before any angle-map or file access and has
three explicit branches: `none`, `rescaled`, and `residual`. `include_amp`
must be a real Boolean. A catch-all residual branch is forbidden.

## Driver and study identity

Public drivers follow `<family>_<verb>_emulator.py`. Family wrappers are thin:
they pin family and program name while using the shared engine for GPU pools,
packing, balancing, journal storage, and sweep logic. A wrong-family YAML
fails at startup and names the correct driver.

A tuning study creates one canonical manifest before workers start. The
manifest includes family, probe, objective and threshold selection, fully
resolved fixed configuration, search-space schema, fixed activation and
rescaling, scientific input digests, companion-file digests, and implementation
identity. Worker count, timeout, GPU count, RAM share, and quiet mode are
operational and excluded.

The manifest and digest are persisted as study attributes. Resume compares
the current manifest before adding work to the queue or starting processes.
Any scientific difference refuses and names changed fields. A journal without
a manifest is refused;
it is never silently blessed. A failed or abandoned earlier trial does not
suppress the known-default control.

Study-name resolution is pure and stable. Direct CosmoLike tuning uses the
established cosmic-shear study name. Each family wrapper has one stable family
tag. An intended rename requires an explicit migration.

Workers return a structured success or failure, not NaN as a scientific
metric. The parent requires one finite success for every requested point
before publishing a normal result and exiting successfully. Missing,
duplicate, non-finite, or failed points produce a nonzero exit. Serial and
parallel paths use the same wrapper and semantics. Worker exit codes belong to
the current invocation; an older successful trial cannot hide current worker
failure.

Every spawned process is validated before launch. Parent cleanup lives in
`finally`: terminate and join siblings, close queues, inspect exit codes, and
use a progress watchdog. A live but blocked process is not proof of progress.

## Diagnostics

Diagnostic eligibility follows the capabilities of the selected loss. A loss
whose score requires parameter values marks the local-linear comparison, which
fits a linear model near each requested point, unavailable while other valid
pages still run. Scalar NPCE therefore produces its supported coverage,
hardness, residual, and PDF
products without calling the incompatible local-linear path.

Local-linear analysis requires positive row counts,
`k_nn > n_parameters + 1`, and `k_nn <= n_train`, where `k_nn` is the number
of nearest training neighbors used for each local fit. The fit also requires
enough distinct reference points. It is one comparator under locality and
smoothness assumptions, not a
mathematical lower bound.

Wide-output diagnostics are bounded. Chi-squared scoring streams. The
local-linear comparison uses bounded coordinate chunks, a declared validation
subsample, or an explicit unavailable status. Grid2d summaries avoid holding
three full float64 copies. Acceptance sets a small memory ceiling on a wide
synthetic problem and still requires the product to finish.

Undefined statistics use structured status, reason, and counts; numeric NaN is
never used as a marker for unavailable state. Coverage reports good and bad
counts. An all-good result
says there are no failures; an all-bad result says comparison is unavailable.
Hard-direction regression checks response variance, feature variance, log
domain, and row sufficiency before fitting. CMB fractional residuals use a
documented spectrum-aware validity mask and per-multipole valid counts; the
error-bar panel remains authoritative at a zero crossing.

Every direct diagnostic chi-squared producer passes through the shared score
domain before statistics or plotting. CMB, grid, and grid2d residual functions
each require a live valid-output test and a live corrupt-score refusal; an AST
search for all call names is only supplemental breadth.

Documentation distinguishes measurement from interpretation. Define nearest-
neighbor distance, Spearman correlation, local linear regression, percentile,
decile, and R-squared at first use. A correlation can be consistent with a
coverage limitation but cannot prove cause. Every heuristic control is a
named argument or constant with units, derivation, and a sensitivity check.

## Publication and plots

Persist a trained artifact before optional diagnostics, so a plotting failure
does not destroy a valid model. Diagnostic logs and products must state
unavailable analyses plainly.

Table writers validate all column lengths before opening a destination.
Unequal values, fractions, labels, or learning-curve lengths raise without
replacing an existing product.

`plot_xi` validates input before creating a figure. When parameters are
provided, one finite min/max normalization and one Matplotlib
`ScalarMappable`, the object that maps numeric values to colors, determine the
exact color reused for each curve and its marker in every panel. Visual
identity is precomputed by curve index. The constant-parameter behavior is
explicit. Without parameters, index colors are allowed and no parameter
colorbar is drawn. Mutable list defaults are forbidden.

Learning-curve and sweep plots share one y-scale decision. Training sizes are
finite and positive; fractions and target are finite in `[0, 1]`. Positive-
only fractions use logarithmic y. Any exact zero uses a zero-capable scale and
the zero marker remains visible at its exact coordinate.

`--quiet` means successful standard output (`stdout`) is empty. All reachable
staging, geometry, training, and worker paths use one explicit output channel.
Errors remain nonzero and go to standard error (`stderr`). Quiet mode never
suppresses persisted scientific metadata.

## Acceptance contracts

The following sections define required evidence without recording who found a
problem, when it was reviewed, or how it moved between branches.

### Loss and schedule acceptance

- Analytic BerHu and capped-BerHu values match reference formulas within
  `1e-9`; join derivatives match within `1e-6`; blend endpoints exactly match
  square root and full BerHu.
- Exact-fit gradients are finite zero in eager and compiled execution. A
  forced compile failure is reported as a failure, never as a pass.
- Decreasing step schedules preserve their established values. Increasing
  zero-to-one schedules have strict interior values and reach one only at the
  endpoint.
- Trim and focus accept constant schedules. BerHu and EMA reject constant
  anneals at the exact dot-separated setting path. Omitted anneals activate
  the full feature.
- Every malformed schedule and range fails before staging. Valid ordinary,
  phase-local, tuning, and sweep paths resolve identically.

### Finite training acceptance

- `eval_val` rejects one NaN, positive infinity, or negative infinity among
  otherwise good rows before ranking and names source positions.
- A NaN scalar loss fails before backward. A finite loss with a non-finite
  gradient fails before optimizer mutation. Weights remain bitwise unchanged.
- Warm-start and transfer parity reject non-finite inputs, outputs, and
  comparisons before printing success or an unrelated parity error.
- Two finite batches near `1e38` produce a finite CPU-float64 epoch mean. A
  mutation restoring device-float32 multiplication overflows and fails.
- Eight finite float32 validation scores near `1e38` produce the direct
  float64 mean, ordinary median, and finite fractions. The returned mean is
  the history value. A float32-mean mutation fails on CPU and the mandatory
  CUDA mirror.
- Every parameter and optimizer-state tensor is finite after a real AdamW
  update. Deliberately poisoned parameters and moments are detected.

### Batch and resource acceptance

- Resident, RAM-streamed, and memory-mapped-file (`memmap`) training execute
  the same `n_train // bs` steps on the same rows for one seed. Only the single global
  tail may be omitted.
- An adversarial safe chunk of `2 * bs - 1` does not drop `bs - 1` rows from
  every chunk. A divisible control uses every row exactly once modulo shuffle.
- Validation with a smaller safe chunk than training never requests more than
  its own limit and scores every row.
- For `out_dim=7`, `target_dim=14`, batch size 3, and float32, packed-target
  arithmetic adds exactly 84 bytes. A crafted budget changes the old two-batch
  estimate to one corrected batch. Four bytes below one complete batch raises.
- Sizing preserves BatchNorm state, every train/eval mode, and CPU/CUDA random
  streams. Mixed-dtype parameters and buffers equal a direct byte sum.
- Token plans allowing four, two, or one concurrent job enforce those counts
  across setup and execution. Moving token acquisition below setup must fail
  the allocation-counter mutation.

### Repeated-run acceptance

- Enabled-to-disabled and disabled-to-enabled loss features match fresh-object
  controls. Failure followed by a valid point also matches a fresh control.
- Two transfer-refinement points enter with the same pristine digest. Reverse
  order and different lane counts produce the same per-point result for fixed
  seeds.
- A point that updates the base and then raises cannot alter the next point's
  entry state. Fixed-geometry and rebuilt-geometry driver paths are both
  exercised.
- Every point's anchor reference matches the pristine source, and every
  correction stage begins with live mode false.

### Study and sweep acceptance

- An identical manifest resumes. Changes to loss, range kind or bound, input
  bytes at the same path, family, activation, rescaling, objective, or source
  identity refuse before workers start.
- A legacy no-manifest journal refuses with a migration instruction.
- Operational-only changes resume.
- A failed-only study still schedules the default control once.
- Direct CosmoLike and every wrapper resolve stable names. Restoring a
  `family is None` conditional must reproduce the study fork and fail.
- Missing, duplicate, NaN, infinity, or failed sweep results prevent normal
  publication. Serial and pooled metadata agree.

### Diagnostic and plotting acceptance

- All-good, all-bad, constant-feature, constant-response, invalid-log-domain,
  and zero-crossing fixtures produce defined status/reason records and no
  formatted NaN.
- Local-linear negative, NaN, and infinite scores refuse before thresholding.
  Values on both sides of the family band normalize or refuse consistently.
- CMB, grid, and grid2d residual diagnostics each execute a finite case with a
  declared expected answer and a corrupt-score refusal. Mutations keep the
  helper call name but pass the wrong tensor, owner, or positions; live tests
  must catch them.
- A wide synthetic diagnostic completes under the declared memory ceiling.
- Parameter-colored lines match the colorbar ScalarMappable under unsorted,
  uneven parameters and retain line/marker identity across panels.
- Positive-only learning curves use log y. A curve ending at zero uses the
  zero-capable scale and displays the zero marker. Invalid fractions and
  training sizes fail before figure construction.

## Stable gate evidence anchors

These anchors are compatibility targets for the structured evidence map. A
gate result is evidence only for the assertion stated here. An exit code or an
exact status output line does not prove an unstated numerical property. A
reference comparison also requires nonempty exact output-line lists; two empty
lists cannot establish equality.

### EMA disabled and EMA smoke

<a id="ema-off-identity-evidence"></a>
**EMA-disabled comparison.** The configured current and reference commands
must read the same declared scientific inputs and compare nonempty exact epoch
output lines after removing only the timing field. Dataset files reached
indirectly through a dataset pointer remain part of scientific identity.

<a id="ema-off-identity-golden-selected-text-equality"></a>
The exact epoch output-line lists from the current and reference commands must
be nonempty and equal after the single documented timing field is removed.

<a id="ema-smoke-evidence"></a>
**EMA smoke.** An EMA-enabled public training command must exit zero, print the
resolved horizon, and reach the rewind path. This is startup and routing
evidence; it does not independently prove which epoch was restored.

<a id="ema-smoke-driver-exit-zero"></a>
The EMA-enabled training subprocess exits with status zero.

<a id="ema-smoke-horizon-banner-present"></a>
The command output contains the exact line `ema: horizon 3 epochs` for the
configured fixture.

<a id="ema-smoke-rewind-line-present"></a>
The command output contains the exact line `rewound to best epoch`. Exact
restored-state truth belongs to the selection-record and snapshot checks.

### Phase resolution and scheduler override

<a id="single-phase-demotion-evidence"></a>
**Single-phase demotion.** The ResMLP command exits zero with a demotion notice,
and a two-phase ResCNN control also exits zero. The control proves reachability,
not output equality.

<a id="single-phase-demotion-single-phase-exit-zero"></a>
The ResMLP command exits with status zero.

<a id="single-phase-demotion-demotion-text-present"></a>
An exact ResMLP output line names single-phase resolution or demotion in plain
language.

<a id="single-phase-demotion-two-phase-control-exit-zero"></a>
The ResCNN two-phase control exits with status zero.

<a id="head-scheduler-override-evidence"></a>
**Head scheduler override.** The override command exits zero and prints the
resolved head-scheduler status line. A reference comparison, when configured,
uses nonempty exact output-line lists. The fixture forces a plateau only in the
30-epoch head phase so the gate can count every learning-rate transition.

<a id="head-scheduler-override-golden-selected-text-equality"></a>
The phase, epoch, and best output-line lists from the current and reference
commands must be nonempty and equal after timing removal.

<a id="head-scheduler-override-driver-exit-zero"></a>
The head-override subprocess exits with status zero.

<a id="head-scheduler-override-override-banner-present"></a>
The command output contains the exact line `[head overrides: scheduler]` for
the fixture.

<a id="head-scheduler-override-lr-cut-cadence"></a>
The gate must parse exactly head epochs 1 through 30. After the eight-epoch
warmup, the learning rate stays fixed through epoch 19, changes once at epoch
20, and then stays fixed through epoch 30. The new value must equal `0.8` times
the pre-cut value. A missing epoch, an early or late cut, or a second cut fails
the assertion.

### Production diagnostic command

<a id="production-diagnostic-evidence"></a>
**Production diagnostic smoke.** A complete repository search, package import,
command exit, and the resolved-size output line are separate assertions.
Expected cut rows, PDF readability, and triangle shading require independent
programmatic checks.

<a id="production-diagnostic-retired-class-name-census"></a>
The complete stated Python-file search has zero matches for the named retired
classes. Exclusions and the filename-pattern scope are part of the assertion.

<a id="production-diagnostic-package-import"></a>
Importing `emulator`, `emulator.designs`, and `emulator.losses` exits zero.

<a id="production-diagnostic-driver-exit-zero"></a>
The public diagnostic training command exits zero.

<a id="production-diagnostic-sizes-banner"></a>
The command output contains a syntactically valid `used N of P cut rows` line.
This assertion does not prove either integer without an independent
calculation.

<a id="production-diagnostic-cut-row-selection"></a>
Cut-row truth requires an independent expected-row calculation from the
fixture and exact comparison with the reported and executed row identities.

<a id="production-diagnostic-diagnostics-pdf"></a>
The requested PDF exists, is nonempty, and can be parsed after command exit.

<a id="production-diagnostic-triangle-shading"></a>
Inspection of the Matplotlib plot objects must compare expected and actual
triangle shading. A visual instruction alone is `UNAVAILABLE`.

### Evaluation partition invariance

<a id="eval-batch-invariance-evidence"></a>
**Evaluation partition invariance.** The same twelve distinct float32 scores
and permuted row identities pass through real source scoring, validation, and
one training epoch under a full batch, equal partitions, and a smaller final
partition. The independent float64 reference is median `1.5`, mean
`3.941666666728755`, and fractions
`[0.833333333, 0.583333333, 0.25]`. The relative tolerance (`rtol`) is `1e-6`,
and the absolute tolerance (`atol`) is `1e-7`, for aggregate metrics; row
identity and order must match exactly.

<a id="eval-batch-invariance-partition-invariance"></a>
Returned score tensors, row positions, median, mean, fractions, histories, and
plateau-scheduler input match the reference in every partition. Dropping a
middle partition or reassigning its scores under unchanged row labels must
fail while the diagnostic reference remains unchanged.

<a id="eval-batch-invariance-ordinary-median"></a>
The real validation path returns the arithmetic midpoint for an even sample,
preserves the odd control, remains partition-invariant, and rejects a
lower-middle `Tensor.median` mutation.

<a id="eval-batch-invariance-cuda-timing"></a>
CUDA timing is evidence only after a reproducible protocol and numerical
acceptance bound are defined. Printed durations alone are `UNAVAILABLE`.

<a id="eval-batch-invariance-production-timing-claim"></a>
A production-speed claim requires a production timing protocol and bound. The
numerical gate makes no speed claim.

<a id="eval-batch-invariance-real-partitions"></a>
The real-entry-point partition test must keep the returned-boundary mutation
and the parameter/target reassociation mutation. Both must fail independently
of the source-score reference.

### Finite-contract compatibility anchors

The following identifiers remain stable for existing evidence links. When a
separately named finite-contract anchor appears later, the entry points to it.
The other entries state their current check directly.

<a id="finite-contract-evidence"></a>
The finite-contract check covers validation, training updates, diagnostics,
warm-start parity, transfer parity, safe square roots, epoch and validation
reductions, score-domain bands, optimizer schema, and post-step state.

<a id="finite-contract-validation-score-finiteness"></a>
See `finite-contract-validation`: non-finite validation rows refuse before
ranking and finite controls preserve metrics.

<a id="finite-contract-train-step-finiteness"></a>
See `finite-contract-train-step`: non-finite loss or gradient refuses before
optimizer mutation.

<a id="finite-contract-diagnostic-score-finiteness"></a>
See `finite-contract-diagnostic`: non-finite diagnostic scores refuse with row
identity.

<a id="finite-contract-finetune-parity-finiteness"></a>
See `finite-contract-finetune-parity`: non-finite warm-start surfaces cannot
produce a success or misleading parity message.

<a id="finite-contract-transfer-parity-finiteness"></a>
See `finite-contract-transfer-parity`: non-finite transfer surfaces refuse
before the frozen-base comparison.

<a id="finite-contract-safe-sqrt-eager"></a>
The exact-fit square-root rule requires eager gradients at zero to be finite
zero and requires invalid score-domain values to be refused.

<a id="finite-contract-safe-sqrt-compiled"></a>
The same exact-fit rule requires a compiled backward pass on a machine that
supports compilation.

<a id="finite-contract-epoch-mean-finiteness"></a>
See `finite-contract-epoch-mean`: CPU float64 accumulation protects the
published epoch mean.

<a id="finite-contract-chi2-domain-boundary"></a>
The chi-squared domain rule must serve training, evaluation, and diagnostics
through one shared validator.

<a id="finite-contract-chi2-width-band"></a>
The allowed roundoff band scales with the number of kept coordinates, not the
square of that width.

<a id="finite-contract-chi2-compute-dtype-band"></a>
The model's compute data type determines machine epsilon for the band; a later
upcast does not change that boundary.

<a id="finite-contract-optimizer-schema"></a>
The optimizer protocol requires every numeric control to be finite and inside
its declared domain before optimizer construction.

<a id="finite-contract-extreme-scale-validation-reduction"></a>
See `finite-contract-extreme-scale-reduction`: finite extreme rows must produce
finite float64 published reductions on CPU and CUDA.

<a id="finite-contract-optimizer-post-step-finiteness"></a>
See `finite-contract-optimizer-post-step`: parameters and optimizer state are
finite after a real update.

### BerHu and schema evidence

<a id="berhu-loss-evidence"></a>
**BerHu transform.** Synthetic float64 probes compare values, join derivatives,
and blend endpoints with independent formulas. A public smoke command checks
configuration routing and exact required output lines. These are separate
assertions.

<a id="berhu-loss-reference-values"></a>
BerHu and capped-BerHu values match independent piecewise references within
absolute error `1e-9` at default and non-default knots.

<a id="berhu-loss-join-derivatives"></a>
Slopes from PyTorch automatic differentiation (`autograd`) on both sides of the
first join and capped second join match analytic derivatives within absolute
error `1e-6`.

<a id="berhu-loss-anneal-endpoints"></a>
Blend zero exactly reproduces square root; blend one exactly reproduces the
configured BerHu transform.

<a id="berhu-loss-golden-selected-text-equality"></a>
When a reference command is configured, the exact output-line lists from the
current and reference commands are nonempty and equal after timing removal.
Otherwise this assertion is `UNAVAILABLE`.

<a id="berhu-loss-smoke-exit-zero"></a>
The configured BerHu public training command exits zero.

<a id="berhu-loss-loss-banners"></a>
The command output contains the resolved `sqrt` and capped-BerHu mode, knot,
and cap for the fixture.

<a id="loss-schema-equivalence-evidence"></a>
**Nested loss schema.** The nested-loss command exits and prints its resolved
BerHu settings. Equivalence to the declared reference configuration requires
a configured reference command and a nonempty exact-output-line comparison;
the status line alone does not prove numerical equivalence.

<a id="loss-schema-equivalence-golden-selected-text-equality"></a>
The exact output-line lists from the current and reference commands are
nonempty and equal after timing removal, or the assertion is `UNAVAILABLE`
when no reference is configured.

<a id="loss-schema-equivalence-smoke-exit-zero"></a>
The nested-loss public command exits zero.

<a id="loss-schema-equivalence-berhu-banner"></a>
An exact command-output line names the resolved capped-BerHu mode, knot `0.2`,
and cap `10`.

### Schedule and EMA evidence

<a id="berhu-anneal-evidence"></a>
**BerHu anneal.** The public command exits and names the resolved hold, length,
and shape. Numerical schedule behavior requires direct endpoint and interior
value assertions.

<a id="berhu-anneal-golden-selected-text-equality"></a>
A configured reference comparison uses nonempty exact output-line lists after
timing removal; otherwise the assertion is `UNAVAILABLE`.

<a id="berhu-anneal-smoke-exit-zero"></a>
The BerHu-anneal public command exits zero.

<a id="berhu-anneal-anneal-banner"></a>
The command output contains the exact line `anneal: hold 5 + 10 cosine` for the
fixture.

<a id="berhu-anneal-schedule-behavior"></a>
The CPU witness calls the production `anneal_value` function for the configured
zero-to-one cosine schedule. Epochs 4, 5, 6, 10, 15, and 16 must return,
respectively, `0`, `0`, `0.024471741852423234`, `0.5`, `1`, and `1` within the
declared numerical tolerance. Probes immediately to the left and right of
epochs 5 and 15 must also establish continuity. A constant schedule or a ramp
that begins one epoch early fails this assertion.

<a id="ema-anneal-evidence"></a>
**EMA anneal.** The command exits and names both the horizon and schedule.
Direct parsing must identify the first epoch whose reported metrics use EMA
weights.

<a id="ema-anneal-golden-selected-text-equality"></a>
A configured reference comparison uses nonempty exact output-line lists after
timing removal; otherwise the assertion is `UNAVAILABLE`.

<a id="ema-anneal-smoke-exit-zero"></a>
The EMA-anneal public command exits zero.

<a id="ema-anneal-ema-anneal-banners"></a>
The command output contains the exact lines `ema: horizon 3 epochs` and
`anneal: hold 5 + 10 cosine` for the fixture.

<a id="ema-anneal-schedule-behavior"></a>
The CPU witness applies the same known-answer values and continuity checks as
`berhu-anneal.schedule-behavior` to the production schedule selected by the EMA
configuration. This proves the shared schedule function and the EMA fixture's
schedule inputs; it does not replace the full training run.

<a id="ema-anneal-live-point-metrics"></a>
The production run must print exactly one `ema first-live` record. It must occur
at epoch 6, carry schedule value `0.024471741852423234`, name a positive number
of optimizer steps, and contain finite raw and averaged validation means and
medians. The gate independently recomputes beta as
`1 - 1 / (3 * schedule * steps_per_epoch)`, with the production zero floor for
a sub-step window, and requires the printed beta to match.

### Joint-training evidence

<a id="joint-training-evidence"></a>
**Joint training.** Joint and frozen-trunk control commands exit and print the
resolved phases. Each command prints one finite phase-boundary digest of the
shared trunk before and after phase 2. The gate compares the digest strings
instead of trusting a printed changed/unchanged label.

<a id="joint-training-golden-selected-text-equality"></a>
A configured reference comparison uses nonempty exact output-line lists after
timing removal; otherwise the assertion is `UNAVAILABLE`.

<a id="joint-training-joint-exit-zero"></a>
The `freeze_trunk: false` command exits zero.

<a id="joint-training-two-phase-banner"></a>
An exact command-output line states the resolved positive number of trunk
epochs.

<a id="joint-training-joint-phase-banner"></a>
An exact command-output line names the joint phase.

<a id="joint-training-control-exit-zero"></a>
The `freeze_trunk: true` control exits zero.

<a id="joint-training-joint-trunk-digest-change"></a>
The `freeze_trunk: false` run must print one phase `joint` record whose before
and after SHA-256 digests differ. The digest covers the trunk parameter names,
dtypes, shapes, and exact bytes in sorted name order and refuses empty or
nonfinite parameter sets. A head-only change cannot satisfy this assertion.

<a id="joint-training-frozen-trunk-digest-identity"></a>
The `freeze_trunk: true` control must print one phase `head` record whose before
and after trunk digests are exactly equal. The parameter count must stay
positive and unchanged. This is parameter-state evidence; it makes no timing
or phase-handoff loss claim.

### Weight-decay evidence

<a id="weight-decay-census-evidence"></a>
**Weight-decay partition.** A small fixed test model made of nested modules is
divided by module role into disjoint decay and no-decay groups whose union
contains each parameter exactly once. This fixture is not a complete search of
every future production model.

<a id="weight-decay-census-allowed-weight-set"></a>
The decayed set is exactly the `Linear`, `Conv1d`, and `BinLinear` weights in
the fixture.

<a id="weight-decay-census-undecayed-role-exclusions"></a>
Activation shape parameters, every bias, affine gains, and unlisted module
parameters remain outside the decayed set.

<a id="weight-decay-census-parameter-group-partition"></a>
The two groups are disjoint and their union contains every fixture parameter
exactly once.

<a id="weight-decay-census-zero-decay-inert"></a>
A requested decay of zero produces zero decay in every optimizer group.

<a id="weight-decay-census-golden-selected-text-equality"></a>
A configured reference comparison uses nonempty exact output-line lists after
timing removal; otherwise the assertion is `UNAVAILABLE`.

### Diagnostic score-domain evidence

<a id="diagnostics-domain-score-boundary"></a>
The shared diagnostic boundary preserves valid scores, converts within-band
roundoff to exact zero, and refuses materially negative, NaN, and infinite
scores while naming producer, rows, and band. The real local-linear and CMB,
grid, and grid2d residual functions each require live valid and corrupt-score
tests. A mutation that bypasses only the boundary must recreate a false
perfect score and fail.

### Canonical finite-contract assertions

<a id="finite-contract-validation"></a>
Real `eval_val` raises on one NaN, positive infinity, or negative infinity
among otherwise valid rows and names the validation side and row. The finite
control reproduces independent mean, ordinary median, and fractions.

<a id="finite-contract-train-step"></a>
Real `training_loop_batched` raises on a NaN scalar loss before backward and
on a non-finite gradient before `optimizer.step`. Model weights remain
bitwise unchanged. A finite control completes and the global-gradient-norm
calculation does not mutate gradients.

<a id="finite-contract-diagnostic"></a>
Real source scoring raises on a non-finite per-row chi-squared value and names
the diagnostic side and source row. A finite control returns unchanged
scores.

<a id="finite-contract-finetune-parity"></a>
Warm-start construction rejects no-extra both-arm NaN, one-arm NaN, infinity,
and extras-present NaN before any success, tolerance, or extras-leak message.
A valid source retains the successful parity result.

<a id="finite-contract-transfer-parity"></a>
Transfer construction rejects a non-finite epoch-zero surface before any
frozen-base comparison. A valid zero-initialized transfer retains the
successful parity result.

<a id="finite-contract-epoch-mean"></a>
A finite per-batch loss near the float32 maximum yields a finite epoch mean
through CPU float64 accumulation. Restoring the device-float32
`loss * batch_size` product must overflow in the mutation case and fail.

<a id="finite-contract-extreme-scale-reduction"></a>
Eight individually finite float32 validation scores near `1e38` publish a
finite float64 mean, ordinary median, and fractions through real validation,
and the returned mean reaches real history append. CPU and CUDA run the same
fixture. Restoring float32 mean reduction must fail where it overflows; a
backend without that overflow reports a control result rather than omitting
the mandatory backend check.

<a id="finite-contract-optimizer-post-step"></a>
After a real AdamW update, every model parameter and optimizer-state tensor is
finite. Deliberately poisoning one parameter and one optimizer moment must be
detected by separate mutations.

## Final review checklist

A training-stack change receives GO only when all applicable answers are yes:

1. Does one source own each formula, state transition, capability, and
   resolved value?
2. Are configuration types, domains, and unknown keys rejected before
   mutation or expensive setup?
3. Are the compute dtype, kept width, row identity, and reduction precision
   explicit at every numerical boundary?
4. Does a non-finite or materially negative score refuse before ranking,
   selection, plotting, or publication?
5. Do I/O chunking, device capacity, worker count, point order, and earlier
   failures leave scientific semantics unchanged?
6. Does every saved record describe the weights and settings that actually
   ran?
7. Does every acceptance test call the real public or owned boundary and use
   an independent expected answer?
8. Does at least one deliberate mutation recreate the prohibited behavior and
   fail the test?
9. Are missing accelerator or compiler checks reported as not passing rather
   than converted to success?
10. Is the note free of dates, personal references, ticket codes, review
    biography, branch diaries, and obsolete status reports?

Any no answer is NO-GO.
