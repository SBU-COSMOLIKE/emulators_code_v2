# Inside the `emulator/` package

An **emulator** is a fast approximation to a slower scientific calculation.
The Python files in this folder define the approximation, train it, check its
errors, and reload the saved result.

For a first training run, read only the three-question **Main guide** below.
It ends at “Where should the reader go next?”; every later section is optional
reference material.

Start with [the main README](../README.md) to choose a task. Use
[the example-YAML guide](../example_yamls/README.md) to prepare and run a
training file. Readers changing the Python package can open the separate
[code reference](CODE_REFERENCE.md).

## Contents

### Main guide

1. [What does this folder do?](#main-purpose)
2. [How does a calculation move through it?](#pipeline-overview)
3. [Where should the reader go next?](#main-next-guide)

### Common questions raised by developers

- **Appendix A — Package and scientific outputs**
  - [Where is each part of the package?](#2-layout)
  - [Which physical quantities can one emulator predict?](#3-the-five-emulator-families)
  - [How does a saved emulator rebuild without guessing?](#faq-package-a3-saved-rebuild)
- **Appendix B — Model design**
  - [What does the residual trunk calculate?](#faq-model-a1-residual-trunk)
  - [What do activation and normalization change?](#faq-model-a2-activations-and-normalization)
  - [How does the convolutional head use an ordered output?](#faq-model-a3-cnn-head)
  - [How does the transformer head use tokens?](#faq-model-a4-transformer-head)
  - [What does the polynomial base contribute?](#faq-model-a5-pce)
  - [How can a run reuse a saved model?](#faq-model-a6-reusing-a-saved-model)
- **Appendix C — Training**
  - [How does the loss turn errors into one number?](#faq-training-a1-loss-transforms)
  - [Which parameters change at each optimizer step?](#faq-training-a2-parameter-updates)
  - [Which settings change during training?](#faq-training-a3-schedules)
  - [How does two-phase training separate the trunk and head?](#faq-training-a4-two-phase-training)
  - [How does the saved record describe several training passes?](#faq-training-a5-pass-record)
- **Appendix D — Code changes**
  - [Which file should a code reader open first?](#code-reader-start)
  - [Where does each file's real work start?](#code-reader-main-functions)
  - [Open the full Python code reference](CODE_REFERENCE.md)

---

## Main guide

### 1. What does this folder do? <a id="main-purpose"></a>

Training starts with rows of cosmological parameters and matching rows from a
scientific calculation. The code learns a function that predicts the second
row from the first. The saved emulator can then replace repeated calls to the
slower calculation in the region where its accuracy has been checked.

This folder contains the reusable Python library. The commands that start a
run are in the repository folder above it.

### 2. How does a calculation move through it? <a id="pipeline-overview"></a>

Here a **neural network** is a mathematical function built from adjustable
numbers called **weights**. Training changes those weights to reduce the
prediction error.

Keep this five-step picture in mind:

```text
scientific training tables
           |
           v
select matching rows and put numbers on comparable scales
           |
           v
predict with a neural network
           |
           v
compare the prediction with validation data
           |
           v
save <family>-<product>-<digest>.emul and the matching .h5
```

**Validation data** are rows kept out of the weight updates and used to check
how the fitted model behaves on examples it did not train on.

The `.emul` file stores the fitted PyTorch weights. The matching `.h5` file
stores the instructions needed to rebuild the model, the coordinate
conversions, and the scientific labels. Keep the two files together.

The readable part of the name identifies the output, such as `cmb-tt` or
`scalar-h0-omegam`. The digest identifies the completed model settings, exact
selected rows, and any saved model reused by the run. Two different scientific
products therefore cannot share a result name merely because they use the same
network and row count. Saving also refuses every name that already has a
`.emul`, `.h5`, symbolic link, or interrupted-save marker; it never replaces
the earlier bytes.

### 3. Where should the reader go next? <a id="main-next-guide"></a>

| Goal | Read |
|---|---|
| choose and run a training command | [main README](../README.md) |
| copy and edit a YAML settings file | [example-YAML guide](../example_yamls/README.md) |
| create training and validation tables | [data-generation guide](../compute_data_vectors/README.md) |
| use a saved emulator in Cobaya | [Cobaya guide](../cobaya_theory/README.md) |
| understand a model or training choice | [Appendix B](#faq-model-a1-residual-trunk) or [Appendix C](#faq-training-a1-loss-transforms) below |
| change this Python package | [Appendix D](#code-reader-start), then the [Python code reference](CODE_REFERENCE.md) |

The appendices are reference material. A first training run does not require
them.

---

## Common questions raised by developers

### Appendix A1. Where is each part of the package? <a id="2-layout"></a><a id="1-layout"></a>

Most changes begin in one of four places:

| Part | Purpose |
|---|---|
| `geometries/` | convert cosmological parameters and scientific outputs to and from model coordinates |
| `designs/` | define the neural-network shapes |
| `losses/` | turn prediction errors into the value minimized during training |
| `training.py` | update model weights and select the saved epoch |

The supporting files have narrower jobs:

| Files | Purpose |
|---|---|
| `parameter_table.py`, `data_staging.py` | read matching rows and keep large arrays in RAM or on disk |
| `batching.py` | move each batch to the processor used for training |
| `experiment.py` | check the YAML choices and assemble one run |
| `cocoa.py` | resolve project, YAML, and output paths |
| `warmstart.py` | prepare fine-tuning and transfer from a saved model |
| `model_recipe.py` | check complete model-building instructions before importing a model class |
| `results.py`, `inference.py` | save, reload, and evaluate an emulator |
| `plotting.py`, `diagnostics.py` | calculate checks and make their figures |
| `scheduling.py`, `family_drivers.py` | divide multi-run searches among devices |
| `background.py`, `syren_base.py`, `analytics.py` | apply analytic physics outside the neural network |

The training commands live in the `driver/` folder beside `emulator/`, not
inside the package. Each driver adds the repository root to its import path
at startup, so the commands run from any working folder.

### Appendix A2. Which physical quantities can one emulator predict? <a id="3-the-five-emulator-families"></a><a id="2-the-five-emulator-families"></a>

The YAML `data` block selects one of five scientific outputs. A **geometry**
is the object that converts an output to and from the numerical coordinates
used by the model. Each family saves its geometry with the fitted weights.

| Family | YAML choice | Prediction | Error measured for one cosmology |
|---|---|---|---|
| cosmic shear | CosmoLike data keys | selected likelihood data vector | $r^{\mathsf T}\Sigma^{-1}r$ |
| scalar | `outputs` | named scalar values | sum of squared standardized residuals |
| CMB spectra | `cmb` | one selected spectrum | sum of squared residuals divided by the saved variance |
| background | `grid` | a function on a redshift grid | sum of squared residuals after the selected coordinate conversion |
| matter power | `grid2d` | a surface on a redshift and wavenumber grid | sum of squared residuals after the selected coordinate conversion |

Here $r$ means prediction minus truth, and $\Sigma^{-1}$ is the inverse
covariance used by the likelihood. All five families return one error number
per cosmology, so they can share the same training loop.

[The example-YAML guide](../example_yamls/README.md#faq-b1-family-blocks)
shows which data block belongs to each family. The remaining appendices here
explain how the common model and training code uses those blocks.

### Appendix A3. How does a saved emulator rebuild without guessing? <a id="faq-package-a3-saved-rebuild"></a>

The `.h5` file stores the information needed to rebuild the predictor:

- The **model recipe** records the exact model class and every constructor
  value used to build it. A constructor is the set of values supplied when a
  Python model object is created.
- The saved **geometries** record how model inputs and outputs are converted.
  When an analytic formula is part of an output, its law is stored with the
  output geometry.
- The **composition mode** states whether the neural-network output stands
  alone or is combined with a saved polynomial or transfer base.

`model_recipe.py` checks the recipe as ordinary data before Python imports the
named model class. It requires every saved field and accepts only known model,
activation, and normalization names. Missing and explicit-null values are
different. For example, `head_act: null` says to inherit the trunk activation,
while a missing `head_act` gives no rebuilding instruction and is refused.

The trusted model constructors and activation factories then check numerical
rules such as positive widths and valid kernel sizes. Saving also compares the
recipe recorded by the live model with the recipe supplied by the experiment.
Reopening loads the learned tensors strictly, so missing, extra, or
wrong-shaped tensors are refused.

`results.py` writes and checks the recipe, saved geometries, and composition
mode. `ai/tests/test_model_recipe.py` covers complete recipes for all six
supported model classes. `ai/tests/test_artifact_recipe_preflight.py` checks
that damaged saved descriptions stop before model loading. A valid
save-and-rebuild check must still reproduce the original prediction.

---

## Appendix B — Questions about model design

### B1. What does the residual trunk calculate? <a id="faq-model-a1-residual-trunk"></a>

The **trunk** is the part of the model used by every model type. It maps the
converted cosmological parameters to the converted scientific output.

```text
converted parameters
        |
        v
linear input layer
        |
        v
repeated residual blocks
        |
        v
linear output layer and fitted scale
        |
        v
converted prediction
```

A residual block calculates a correction and adds the block's input back to
it. This skip connection lets the block learn a small change rather than a
complete replacement. `ResMLP` contains only this trunk. `ResCNN` and
`ResTRF` add a second part, called a **head**, after it.

The YAML values for trunk width and depth are explained in
[the model-settings question](../example_yamls/README.md#faq-e6-model-settings).
The implementation is in `designs/plain.py` and `designs/blocks.py`.

### B2. What do activation and normalization change? <a id="faq-model-a2-activations-and-normalization"></a>

An **activation** is the nonlinear function applied between fitted linear
layers. Without one, several linear layers would collapse into one linear
map. `activations.py` provides the learnable `H`, `power`, `multigate`, and
`gated_power` choices, plus fixed `relu` and `tanh` choices.

**Normalization** changes the scale and offset of hidden features inside the
trunk. The accepted names are `affine`, `per_feature`, and `none`.

`affine` fits one scale and offset for a layer. `per_feature` fits a separate
pair for every hidden feature. `none` applies no such step.

All six activation families may be used inside a plain `ResMLP` trunk. A CNN
or transformer correction head has one extra restriction. Its final learned
layer begins at zero so that a newly enabled head does not change the trunk's
prediction. The activation immediately after that layer must still transmit a
learning signal at zero. The head therefore accepts `H`, `multigate`, and
`tanh`, but currently refuses `relu`, `power`, and `gated_power`.

This does not ban ReLU from the trunk. A two-phase run may use ReLU in the
trunk and pin its head to `H`, `multigate`, or `tanh`. The model-settings guide
shows [the exact startup rules and a Boolean example](../example_yamls/README.md#what-startup-refuses).

These choices change how the model is fitted; they do not change the physical
meaning of the saved output. Their YAML form is in
[the example-YAML guide](../example_yamls/README.md#faq-e6-model-settings).

### B3. How does the convolutional head use an ordered output? <a id="faq-model-a3-cnn-head"></a>

`ResCNN` first uses the residual trunk. It then places the prediction in the
physical output order and slides a one-dimensional convolution along that
ordered coordinate. Nearby angles, multipoles, redshifts, or wavenumbers can
therefore contribute to a local correction.

For cosmic shear, tomographic bins become channels of the convolution. For
CMB and background outputs, the ordered coordinate is multipole or redshift.
For matter power, each redshift slice contains an ordered wavenumber axis.

Some cosmic-shear bins keep different angular positions after masking. The
head records the original slot of every kept value; it does not infer a slot
from how many values survived. A Boolean mask marks the extra rectangle cells
that exist only for storage. The CNN reapplies that mask after each
convolution, activation, and parameter-dependent FiLM shift, so an artificial
cell cannot carry a value into a physical output. Both the slot map and mask
are stored with a trained model. Saving refuses a model and geometry that
describe different layouts, before it changes either output file. Reopening
checks the two saved files against each other again.

Named scalar outputs have no physical neighbor order. The scalar family
therefore refuses `ResCNN` and uses `ResMLP`.

An outer gate controls the size of the CNN correction. The correction itself
still starts at exactly zero, but the gate must start at a finite nonzero
32-bit value so the first update can reach the head. Startup therefore refuses
`gate_init: 0` and values so small that they round to zero, such as `1e-50`.

### B4. How does the transformer head use tokens? <a id="faq-model-a4-transformer-head"></a>

`ResTRF` also begins with the residual trunk. Its head groups the ordered
output into **tokens**, which are small vectors that the transformer compares
with one another. A token represents a tomographic bin for cosmic shear or a
stored coordinate group for CMB, background, and matter-power outputs.

The attention calculation lets one token use information from another token.
The head then adds its correction to the trunk prediction. Named scalar
outputs have no defined token order, so the scalar family refuses `ResTRF`.

The transformer uses the same saved slot map and Boolean mask as the CNN. Its
normalization ignores storage-only coordinates, attention excludes keys and
values with no shared physical coordinate, and a completely masked token
stays exactly zero. This matters for unequal cosmic-shear bins and for the
short final token created when `n_tokens` does not divide an ordered axis
evenly.

The transformer blocks live in `designs/blocks.py`; the complete model lives
in `designs/plain.py`.

The transformer correction uses the same zero-start rule as the CNN
correction. Its outer gate must be finite and remain nonzero when stored as a
32-bit number. The program checks that value before model construction, then
checks `n_heads` against the real token width after the output layout is
known.

### B5. What does the polynomial base contribute? <a id="faq-model-a5-pce"></a>

PCE means **polynomial chaos expansion**. This option first fits a sum of
Legendre polynomials to the converted training targets. The fitted polynomial
is frozen. The neural model then learns the residual left by that base or,
for an allowed CosmoLike setup, a ratio to the base.

The fit checks each output pattern by leaving out one training row at a time.
It keeps a pattern only when that estimated error is finite and strictly below
`pce.loo_max`. If every pattern misses the limit, the run stops before neural
training and before a new emulator file is written. The error message shows
the limit, the best measured error, and the patterns that were tried.

The saved input bounds and polynomial coefficients use `float32`, the number
format used during emulator prediction. The fit builds its polynomial and
repeats the error calculation with those stored values and all retained output
patterns together. A fit therefore cannot pass in higher precision or in a
one-pattern calculation and cross the limit when saved.

A singular-value decomposition (SVD) rewrites a matrix as ordered numerical
directions. The polynomial fit uses a dense matrix, which stores every
element. A matter-power PCE run must fit the thinned target and the SVD matrix
inside their separate memory budgets.

The exact YAML block, supported forms, and family restrictions are in
[the PCE question](../example_yamls/README.md#faq-c4-pce). The polynomial fit
is implemented in `designs/pce.py`; its combinations with neural predictions
are in `losses/pce.py`.

### B6. How can a run reuse a saved model? <a id="faq-model-a6-reusing-a-saved-model"></a>

There are four different actions:

| Goal | Action |
|---|---|
| make predictions without more training | load the `.emul` and `.h5` pair with `EmulatorPredictor` |
| continue fitting a compatible model | use fine-tuning; the inherited weights may all change |
| keep a compatible base fixed and learn a correction | use ordinary transfer; the base weights remain frozen |
| jointly adjust a CosmoLike base after its correction is trained | add the optional transfer `refine` stage |

Fine-tuning is available for all five output families. Its source must be a
plain saved model, not a saved PCE or transfer composition. Transfer is
available for cosmic shear, CMB with no amplitude law, background, and matter
power. Scalar outputs use fine-tuning instead.

Only cosmic-shear transfer may use `refine`; CMB, background, and matter-power
transfer keep the base frozen.

`warmstart.py` checks the source, extends the input conversion when new
parameters are added, and checks the initial prediction before training.
[The YAML guide](../example_yamls/README.md#faq-c2-reuse) gives the
fine-tuning settings.

---

## Appendix C — Questions about training

### C1. How does the loss turn errors into one number? <a id="faq-training-a1-loss-transforms"></a>

The output-conversion code first produces one non-negative error $c$ for each
cosmology in a batch. A **batch** is a small group of rows used for one weight
update. The loss may discard a configured fraction of the largest errors,
apply one of the transforms below, weight the remaining rows, and take their
mean.

| Mode | Transform of $c$ | Effect |
|---|---|---|
| `chi2` | $c$ | gives the largest errors the strongest weight |
| `sqrt` | $\sqrt{c}$ | reduces the influence of very large errors |
| `sqrt_dchi2` | $\sqrt{1+2c}-1$ | is smooth at zero and grows like a square root in the tail |
| `berhu` | square root below one knot, then a linear function of $c$ | gives errors above the knot more influence |
| `berhu_capped` | `berhu` up to a second knot, then a square-root-shaped tail | bounds the derivative of the largest errors |

`trim` drops the largest rows before the mean. `focus` increases the weight
of hard rows that remain. Validation never trims rows. The exact YAML blocks
are in [the loss question](../example_yamls/README.md#faq-e3-loss) and
[the trim, focus, and averaging question](../example_yamls/README.md#faq-e5-trim-focus-ema).

### C2. Which parameters change at each optimizer step? <a id="faq-training-a2-parameter-updates"></a>

One optimizer step reads a small group of training rows, computes the loss,
finds its derivatives with respect to trainable model parameters, and updates
those parameters with AdamW.

Weight decay applies only to the `.weight` parameters of `Linear`, `Conv1d`,
and `BinLinear` layers. Biases, fitted scales, activation parameters,
normalization parameters, and all other parameters receive no weight decay.
A frozen base or a frozen trunk receives no derivative and does not change
during that step.

The learning rate scales with the square root of the batch size relative to
the configured reference batch. [The parameter-update question in the YAML
guide](../example_yamls/README.md#faq-e4-parameter-updates) shows the settings.

### C3. Which settings change during training? <a id="faq-training-a3-schedules"></a>

Some values are allowed to change from one epoch to the next. An **epoch** is
one pass through the selected training rows. The **validation median** is the
median error across the rows kept out of training.

| Setting | What changes |
|---|---|
| learning-rate warmup | starts with smaller weight updates and reaches the configured learning rate |
| plateau scheduler | lowers the learning rate after the validation median stops improving for the configured patience |
| trim schedule | changes the fraction of largest batch errors omitted from the training mean |
| focus schedule | changes how strongly the loss weights hard rows that remain |
| EMA | after its delayed start, forms an exponential moving average of model weights for validation and saved-model selection |
| gradient clipping | limits the norm of one update when `clip` is greater than zero |
| rewind | restores the best saved optimizer and model state when the plateau scheduler lowers the learning rate |

The training loop selects the epoch with the smallest validation fraction
above the first configured $\Delta\chi^2$ threshold. EMA starts only after
learning-rate warmup and once any configured EMA schedule becomes positive.
Linear, cosine, and step schedules can do this after their hold; a constant
zero schedule never starts averaging.

Selection uses the live weights before EMA starts and the averaged weights
afterwards. The plateau scheduler always reads the raw validation median.

The corresponding YAML blocks are grouped in
[the run-controls appendices](../example_yamls/README.md#faq-e2-run-controls).

### C4. How does two-phase training separate the trunk and head? <a id="faq-training-a4-two-phase-training"></a>

Two-phase training applies only to a model with a CNN or transformer head.
`trunk_epochs` gives the number of epochs in the first phase and must be less
than the total `nepochs`.

```text
train the trunk while bypassing the head
                 |
                 v
restore the best trunk epoch
                 |
                 v
start a fresh optimizer and learning-rate schedule
                 |
                 v
train the head with the trunk frozen
or train trunk and head together when freeze_trunk is false
```

The head starts as a zero correction, so enabling it does not create a jump
when training switches phases. The `trunk` and `head` YAML blocks may give
each phase its own learning rate, scheduler, loss, trim, focus, clipping,
rewind, and EMA settings.

[The two-phase YAML question](../example_yamls/README.md#faq-e7-two-phase-training)
contains a complete settings block. `training.py` owns the two training
passes; the model's `set_train_phase` method selects which parameters are
trainable.

### C5. How does the saved record describe several training passes? <a id="faq-training-a5-pass-record"></a>

A **training pass** here is one configured part of a run, not one epoch. A
simple model has one pass. A two-phase model has a trunk pass followed by a
head pass. Transfer refinement adds one final pass in which the saved base and
its correction are adjusted together.

Each saved pass records its epoch count and the settings actually used,
including learning rate, scheduler, loss, trimming, focus, clipping, rewind,
and exponential moving average. It also records which rows of the saved
training curves belong to that pass.

For example, two trunk epochs followed by three head epochs produce:

```text
trunk history rows  [0, 2)
head history rows   [2, 5)
total_epochs        5
```

The brackets mean that row 0 is included and row 2 begins the next pass.
`training.py` creates this ordered record so a later reader can understand
which settings produced each part of a training curve.

This record is provenance, not an instruction for prediction. Before saving,
`results.py` performs one simple check: the loss and validation curves must be
finite arrays with compatible shapes. Reopening an emulator does not read the
history group or interpret the pass plan; it needs the model recipe,
geometries, composition facts, and learned weights instead.

`ai/tests/test_training_pass_recipe.py` checks the record produced by ordinary,
two-phase, and transfer-refinement training. The preflight test checks the
writer's array-shape refusal and proves that a saved model still reopens after
its optional history group is removed.

---

## Appendix D — Which file should a code reader open first? <a id="code-reader-start"></a>

Start with the file in the right column. The
[Python code reference](CODE_REFERENCE.md) gives the longer change map,
model-variant table, and per-file list of functions and classes.

| Task | First file or folder |
|---|---|
| change parameter or output coordinate conversions | `geometries/` |
| change a neural-network shape | `designs/` |
| add or change an activation function | `activations.py` |
| change how prediction errors are combined | `losses/` |
| change which rows enter training | `data_staging.py` |
| change batch loading or device-memory choices | `batching.py` |
| change weight updates or training schedules | `training.py` |
| change how YAML settings build a run | `experiment.py` |
| change fine-tuning or transfer from a saved model | `warmstart.py` |
| change what a saved model contains | `results.py` |
| change predictions made from a saved model | `inference.py` |
| change figures or numerical checks | `plotting.py` or `diagnostics.py` |

### Appendix D2. Where does each file's real work start? <a id="code-reader-main-functions"></a>

Every file in this package has one or two functions or classes where
its real work happens. Everything else in the file prepares inputs for
those few names or checks their results, so a reader who knows the main
names can treat the rest as helpers.

To read the package for the first time, follow one training run through
four files, opening the named function in each:

1. `experiment.py` — `EmulatorExperiment.from_yaml` builds one
   configured run from a YAML file, and `EmulatorExperiment.run` walks
   it end to end: stage the rows, build the conversions, train.
2. `training.py` — `run_emulator` assembles the model, optimizer,
   scheduler, and loaders, then hands them to `training_loop_batched`,
   the loop that updates weights once per epoch.
3. `results.py` — `save_emulator` writes the two-file saved emulator,
   and `rebuild_emulator` reconstructs the model from those files
   alone.
4. `inference.py` — `EmulatorPredictor.predict` answers one cosmology
   from the saved files.

The table below gives the same one-or-two main names for every other
file. A class name means the class is the unit to read, starting at the
named method or its constructor.

| File | Main names | The job they anchor |
|---|---|---|
| `parameter_table.py` | `resolve_parameter_table` | decide which parameters a run reads, in which order |
| `data_staging.py` | `stage_source`, `load_source` | open the generated tables, apply cuts, keep rows in RAM or on disk |
| `batching.py` | `build_loaders` | build the `load(rows)` closures that feed batches to the device |
| `cocoa.py` | `resolve_cocoa_config`, `cocoa_output` | resolve project, YAML, and output paths |
| `validation.py` | `require_exact_int` and its four siblings | one family of shared type-refusal helpers; no single owner |
| `fixed_facts.py` | `parse_sidecar`, `check_support` | read the scientific record; refuse points outside the trained region |
| `model_recipe.py` | `validate_model_recipe`, `check_model_matches_recipe` | check saved model-building instructions before any class is imported |
| `warmstart.py` | `load_source`, `build_warm_start` | open a saved source emulator and move its weights into a new run |
| `activations.py` | `make_activation` | build the activation family a YAML names |
| `designs/plain.py` | `ResMLP` | the baseline trunk model; `ResCNN` and `ResTRF` add heads to it |
| `designs/blocks.py` | `ResBlock`, `TRFBlock` | the shared pieces the models are assembled from |
| `designs/ia.py` | `TemplateMLP` | the factored intrinsic-alignment designs; its CNN and TRF variants mirror `plain.py` |
| `designs/pce.py` | `PCEEmulator.from_training` | fit the frozen polynomial base; `forward` evaluates it |
| `geometries/parameter.py` | `ParamGeometry.encode` | convert raw parameters to model inputs; `decode` inverts |
| `geometries/output.py` | `DataVectorGeometry`, `build_shear_angle_map` | convert data vectors to model targets; attach the per-bin layout |
| `geometries/scalar.py` | `ScalarGeometry` | the same conversion for named scalar outputs |
| `geometries/cmb.py` | `CmbDiagonalGeometry` | the same conversion for CMB spectra, with the amplitude law |
| `geometries/grid.py` | `GridGeometry` | the same conversion for background functions on a redshift grid |
| `geometries/grid2d.py` | `Grid2DGeometry` | the same conversion for matter-power surfaces on (z, k) |
| `losses/core.py` | `make_chi2`, `CosmolikeChi2` | build a run's loss; the covariance-weighted error |
| `losses/scalar.py` | `make_scalar_chi2` | the scalar family's loss |
| `losses/cmb.py` | `make_cmb_chi2` | the CMB family's loss and roughness penalty |
| `losses/ia.py` | `TemplateFactoredChi2` | combine factored templates inside the loss |
| `losses/pce.py` | `PCEResidualChi2` | score the network and the frozen polynomial base together |
| `losses/transfer.py` | `TransferChi2` | score the correction and the frozen base together |
| `background.py` | `distance_interpolators` | distances derived from H(z) |
| `syren_base.py` | `base_pklin`, `base_boost` | the analytic matter-power base formulas |
| `analytics.py` | `analytic_shape_ratio`, `rescale_xi` | the analytic rescaling applied outside the network |
| `scheduling.py` | `run_gpu_pool`, `lpt_assign` | run independent trainings across several GPUs |
| `family_drivers.py` | `read_sweep_block`, `resolved_sweep_record` | parse a sweep's settings and record what it ran |
| `plotting.py` | `plot_diagnostics`, `plot_history` | draw the diagnostic and training-history figures |
| `diagnostics.py` | `coverage_diagnostic`, `local_linear_floor` | compute the numerical checks those figures draw |

The [Python code reference](CODE_REFERENCE.md) lists every public
function and class per file. For a guided study order with exercises,
the code guide (`documentation/emulator_code_guide.tex`) ends with a
file-by-file route through the same files.
