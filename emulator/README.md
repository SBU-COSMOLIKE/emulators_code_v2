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
- **Appendix D — Code changes**
  - [Which file should a code reader open first?](#code-reader-start)
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
save <root>.emul and <root>.h5
```

**Validation data** are rows kept out of the weight updates and used to check
how the fitted model behaves on examples it did not train on.

The `.emul` file stores the fitted PyTorch weights. The matching `.h5` file
stores the instructions needed to rebuild the model, the coordinate
conversions, and the scientific labels. Keep the two files together.

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
| `results.py`, `inference.py` | save, reload, and evaluate an emulator |
| `plotting.py`, `diagnostics.py` | calculate checks and make their figures |
| `scheduling.py`, `family_drivers.py`, `studies/` | divide multi-run searches among devices and record each search |
| `background.py`, `syren_base.py`, `analytics.py` | apply analytic physics outside the neural network |

The training commands sit beside the `emulator/` folder, not inside it. A
command can therefore import this package without an extra Python-path step.

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
