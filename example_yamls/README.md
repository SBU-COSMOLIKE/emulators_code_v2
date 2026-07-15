# Choose and edit an example YAML

Use this folder when you need settings for model training or for the CMB
covariance calculation. An emulator is a trained, fast approximation to a
slower physics calculation. YAML is an indented text file that gives a
program its settings. Each file here is a template: copy the closest one into
your CoCoA project, then edit the copy.

The filenames inside these templates are examples. The training arrays,
parameter tables, and saved models are not stored in this folder.

```mermaid
flowchart LR
  A["Choose what to predict"] --> B["Copy the closest file"]
  B --> C["Change data and run settings"]
  C --> D["Check the YAML syntax"]
  D --> E["Run the matching program"]
```

In words: choose the physical quantity first, copy its example, change the
copy, check that Python can read it, and run the program named in the table
below.

## Contents

### Main guide

1. [Choose a starting file](#choose-file)
2. [Copy it into your project](#copy-file)
3. [Change your copy in a fixed order](#edit-file)
4. [Check the YAML syntax](#check-file)
5. [Run the matching program](#run-file)

### Common questions raised by developers

- [Appendices about YAML structure](#appendix-a-yaml)
  - [How do indentation and comments work?](#faq-a1-yaml-basics)
  - [What are the main blocks in a trainer YAML?](#faq-a2-blocks)
  - [Why do two different settings use the word covariance?](#faq-a3-covariance)
- [Appendices about physical families](#appendix-b-families)
  - [Which part of `data` selects the physical family?](#faq-b1-family-blocks)
  - [How do I make the missing background or matter-power partner?](#faq-b2-partner-files)
- [Appendices about special runs](#appendix-c-special-runs)
  - [What is the difference between a sweep and tuning?](#faq-c1-sweep-tune)
  - [What is the difference between fine-tuning and transfer?](#faq-c2-reuse)
  - [How do I create the CMB covariance file?](#faq-c3-cmb-covariance)
- [Appendices about paths and errors](#appendix-d-paths)
  - [Where does each filename point?](#faq-d1-paths)
  - [What should I check when a run stops at startup?](#faq-d2-startup-errors)

---

## 1. Choose a starting file <a id="choose-file"></a>

A training program is called a *driver* in the code and in the main README.
It reads one YAML, checks the settings it knows, loads the named data, and
starts the requested calculation.

This guide calls each physical output type and its required file layout a
*family*. The five families are cosmic shear, named scalar values, CMB,
background expansion, and matter power.

Choose by the result you want:

| Your goal | Copy this file | Run this program |
| --- | --- | --- |
| Train one cosmic-shear emulator | [`cosmic_shear_train_emulator.yaml`](cosmic_shear_train_emulator.yaml) | `cosmic_shear_train_emulator.py` |
| Predict named values such as `H0` and `omegam` | [`scalar_emulator.yaml`](scalar_emulator.yaml) | `scalar_train_emulator.py` |
| Train one CMB spectrum: TT, TE, EE, or lensing potential | [`cmb_emulator.yaml`](cmb_emulator.yaml) | `cmb_train_emulator.py` |
| Train the supernova-range $H(z)$ model | [`baosn_hubble_emulator.yaml`](baosn_hubble_emulator.yaml) | `baosn_train_emulator.py` |
| Train the nonlinear matter-power boost | [`mps_boost_emulator.yaml`](mps_boost_emulator.yaml) | `mps_train_emulator.py` |
| Try a stated list of values for one cosmic-shear setting | [`cosmic_shear_sweep_hyperparam_emulator.yaml`](cosmic_shear_sweep_hyperparam_emulator.yaml) | `cosmic_shear_sweep_hyperparam_emulator.py` |
| Search several numeric cosmic-shear settings | [`cosmic_shear_tune_emulator.yaml`](cosmic_shear_tune_emulator.yaml) | `cosmic_shear_tune_emulator.py` |
| Continue a saved cosmic-shear model on new data | [`cosmic_shear_finetune_emulator.yaml`](cosmic_shear_finetune_emulator.yaml) | `cosmic_shear_train_emulator.py` |
| Keep a saved cosmic-shear model fixed and learn a correction | [`cosmic_shear_transfer_emulator.yaml`](cosmic_shear_transfer_emulator.yaml) | `cosmic_shear_train_emulator.py` |
| Calculate the covariance used by CMB training | [`cmb_covariance_lcdm.yaml`](cmb_covariance_lcdm.yaml) | `compute_data_vectors/compute_cmb_covariance.py` |

The last row is not emulator training. It produces the `.npz` covariance file
named inside `cmb_emulator.yaml`.

The training-size sweep programs reuse the ordinary YAML for their physical
family. The cosmic-shear comparison of activation functions, the nonlinear
functions used inside the model, also reuses
`cosmic_shear_train_emulator.yaml`. Neither task needs another YAML file.

The $H(z)$ and matter-power examples each cover one member of a two-model
pair. [FAQ B2](#faq-b2-partner-files) explains how to make the other member.

## 2. Copy it into your project <a id="copy-file"></a>

Run the following commands from `$ROOTDIR`, the CoCoA folder set by
`source start_cocoa.sh`. They create a project settings folder and copy the
cosmic-shear example to `my_cosmic_shear.yaml`. They do not change the
repository copy.

```bash
cd "$ROOTDIR"
D=external_modules/code/emulators_code_v2
PROJECT=projects/lsst_y1
CONFIG_DIR=emulators/training_scripts

mkdir -p "$PROJECT/$CONFIG_DIR"
cp "$D/example_yamls/cosmic_shear_train_emulator.yaml" \
  "$PROJECT/$CONFIG_DIR/my_cosmic_shear.yaml"
```

After a successful copy, this file exists:

```text
$ROOTDIR/projects/lsst_y1/emulators/training_scripts/my_cosmic_shear.yaml
```

Replace the source filename and the new filename when you choose another row
from the table. Keep your edited YAML in the folder passed as `--fileroot`.

## 3. Change your copy in a fixed order <a id="edit-file"></a>

Do not read the 470-line cosmic-shear example from top to bottom before
starting. Most of its lines begin with `#`, so they are explanations or
settings that are currently off. The uncommented lines are the active run.

Edit the active settings in this order:

Training rows are examples the program learns from. Validation rows are
separate examples used to check the model while it learns.

| Order | What to change | What it controls |
| --- | --- | --- |
| 1 | The filenames in `data` and the family block such as `cmb`, `grid`, or `grid2d` | Which generated data and physical quantity the run reads |
| 2 | `data.n_train`, `data.n_val`, and `data.split_seed` | How many training and validation rows are selected, and how they are shuffled |
| 3 | `train_args.model` | The learned model's type and size; keep `name: resmlp` for the first run. A fine-tune run omits this block and inherits the saved model. |
| 4 | `train_args.nepochs`, `train_args.bs`, `train_args.loss`, `train_args.lr`, and `train_args.scheduler` | Full passes through the training rows, rows per model update, the error measure, update size, and later learning-rate changes |
| 5 | A special block only when the task needs one | Top-level `sweep`, `pce` (a polynomial base), or `transfer`; or `train_args.finetune`. [FAQ A2](#faq-a2-blocks) shows their locations. |

For your first run, leave the `optimizer`, `trim`, and `focus` blocks
unchanged. They are advanced controls for how the model changes and how the
run treats difficult rows. The training setup reads them even when trimming
and focus are set to zero.

`data.param_cuts` can remove rows outside chosen cosmological-parameter
ranges. `n_train` and `n_val` count rows after those filters. The run stops
when fewer usable rows are available than the requested count.

For cosmic shear, `cosmolike_data_dir` selects a directory under
`$ROOTDIR/external_modules/data`, and `cosmolike_dataset` selects a dataset
inside that directory. The other families have an explicit family block.
[FAQ B1](#faq-b1-family-blocks) lists the selection rules.

Before running CMB, background, or matter-power training, also read
[FAQ D1](#faq-d1-paths). Supporting filenames inside those family blocks are
read from the program's starting folder. The standard CoCoA startup makes the
generated files available there.

## 4. Check the YAML syntax <a id="check-file"></a>

This command checks that the active CoCoA Python environment can read the
file and that its top level is a named group of settings. It does not change
the file. Run it from `$ROOTDIR` after the copy above:

```bash
cd "$ROOTDIR"
PROJECT=projects/lsst_y1
CONFIG_DIR=emulators/training_scripts
YAML_PATH="$PROJECT/$CONFIG_DIR/my_cosmic_shear.yaml"
python - "$YAML_PATH" <<'PY'
from pathlib import Path
import sys
import yaml

path = Path(sys.argv[1])
config = yaml.safe_load(path.read_text())
if not isinstance(config, dict):
    raise TypeError(f"{path}: top level must contain named blocks")
print(f"YAML syntax OK: {path}")
print("top-level blocks:", ", ".join(config))
PY
```

Expected result for the copied training file:

```text
YAML syntax OK: projects/lsst_y1/emulators/training_scripts/my_cosmic_shear.yaml
top-level blocks: data, train_args
```

This check proves only that the YAML syntax can be read. It does not detect a
repeated key, an unsupported setting, a missing data file, or a physical
mismatch between files. PyYAML keeps the last value when one key appears
twice, so remove the old copy of a setting instead of adding a second one.

There is no `--validate-only` or `--check-config` option. The matching program
checks different settings and files while preparing the run. It does not stop
after checking; it proceeds into data loading and training.

## 5. Run the matching program <a id="run-file"></a>

The next command is not a preview. It starts a real training run and may take
a long time. For the copied cosmic-shear YAML, run:

```bash
cd "$ROOTDIR"
D=external_modules/code/emulators_code_v2
PROJECT=projects/lsst_y1
CONFIG_DIR=emulators/training_scripts

python "$D/cosmic_shear_train_emulator.py" \
  --root "$PROJECT" \
  --fileroot "$CONFIG_DIR" \
  --yaml my_cosmic_shear.yaml \
  --save my_cosmic_shear
```

The program prints the selected computer device, model, row counts, and
training settings before the epoch log. Read that startup summary. It shows
the settings the program will use after command-line choices are applied.

The trained model is written as a matching `.emul` and `.h5` pair under
`$ROOTDIR/$PROJECT/chains`. The main README explains the output and optional
diagnostic PDF in [Run and validate](../README.md#start-run).

For another family, keep the same three path options and replace the program
and YAML name with the pair in [the chooser table](#choose-file). For example,
`python "$D/cmb_train_emulator.py" --help` prints the CMB trainer's options.
`--help` does not open or check your YAML.

---

# Common questions raised by developers

# Appendices about YAML structure <a id="appendix-a-yaml"></a>

## FAQ A1. How do indentation and comments work? <a id="faq-a1-yaml-basics"></a>

YAML uses spaces at the start of a line to show which settings belong
together. The examples use two spaces for each level. Do not use tab
characters.

```yaml
train_args:
  model:
    name: resmlp
    mlp:
      width: 128
      n_blocks: 4
```

Here, `model` belongs to `train_args`, and `width` belongs to `mlp`. A name
with no indentation, such as `data` or `train_args`, starts a top-level block.

A line beginning with `#` is a comment and is ignored by the YAML reader.
When you enable a commented block, remove `#` from every needed line while
preserving its indentation.

Use lowercase `true`, `false`, and `null` for switches and an explicit empty
value. Lists put one `-` before each item. Never leave two active copies of
the same key in one block because the later value silently wins.

## FAQ A2. What are the main blocks in a trainer YAML? <a id="faq-a2-blocks"></a>

Every trainer YAML contains these two top-level blocks:

| Block | Meaning |
| --- | --- |
| `data` | Input filenames, requested row counts, shuffle seed, parameter-range filters, and the family-specific output description |
| `train_args` | Epochs, batch size, loss, model, optimizer (the rule used to update it), learning rate, scheduler, trimming, and focus settings |

Some runs add one more block or one block inside `train_args`:

| Setting | Where it goes | Use it for |
| --- | --- | --- |
| `sweep` | Top level | Trying a stated list of values for one setting |
| `pce` | Top level | Fitting a polynomial base before the neural correction |
| `transfer` | Top level | Keeping a saved base model and training a correction |
| `finetune` | Inside `train_args` | Continuing training from a saved model |

The covariance file is different. It uses `theory`, `params`, and `cov_args`
because it runs a CMB calculation rather than emulator training.

The main README gives the complete setting reference beginning with
[What does a complete YAML file contain?](../README.md#2-the-yaml-file).

## FAQ A3. Why do two different settings use the word covariance? <a id="faq-a3-covariance"></a>

`data.train_covmat` is a parameter covariance text file. Its header names and
orders the cosmological input parameters, and its numbers describe their
joint spread.

`data.cmb.covariance` is an `.npz` file for the CMB output spectrum. It holds
the spectrum error information used to scale the training target and measure
prediction error. Create it with the separate covariance program described in
[FAQ C3](#faq-c3-cmb-covariance).

# Appendices about physical families <a id="appendix-b-families"></a>

## FAQ B1. Which part of `data` selects the physical family? <a id="faq-b1-family-blocks"></a>

The scalar, CMB, background, and matter-power families each have an explicit
key inside `data`. Cosmic shear is selected when none of those four keys is
present. Do not combine two family keys in one training YAML.

| Family | Selecting key | What one saved model predicts |
| --- | --- | --- |
| Cosmic shear | None of `outputs`, `cmb`, `grid`, or `grid2d`; it then requires `cosmolike_data_dir` and `cosmolike_dataset` | One ordered set of selected CosmoLike observables |
| Named scalar values | `outputs` | The named columns listed there |
| CMB | `cmb` | One of `tt`, `te`, `ee`, or `pp` |
| Background expansion | `grid` | One function on one redshift grid |
| Matter power | `grid2d` | One surface on a $(z,k)$ grid |

The matching program refuses a YAML that names another family's key. The
main README explains the scientific inputs in
[Appendices about physical families](../README.md#appendix-c-families).

For scalar training, every name in `outputs` must be a column in the parameter
table's `.paramnames` companion file. For CMB training, changing `spectrum`
also requires matching training and validation spectrum arrays and a
covariance with the same multipole grid.

## FAQ B2. How do I make the missing background or matter-power partner? <a id="faq-b2-partner-files"></a>

The shipped background file trains $H(z)$ on the supernova redshift range.
A complete background service also needs a second saved model for the
recombination-range distance $D_M$. Copy the file again, point it at the
`D_M` arrays and redshift file, and change these settings:

| `data.grid` setting | $H(z)$ copy | $D_M$ copy |
| --- | --- | --- |
| `quantity` | `Hubble` | `D_M` |
| `units` | `km/s/Mpc` | `Mpc` |
| `law` | `log_offset` | `none` |
| `offset` | Required | Remove it |

The shipped matter-power file trains the nonlinear boost
$P_{\rm nl}/P_{\rm lin}$. A complete matter-power service also needs a second
saved model for linear $P(k,z)$. Copy the file again, point it at the linear
arrays and base files, and change:

| `data.grid2d` setting | Boost copy | Linear-$P$ copy |
| --- | --- | --- |
| `quantity` | `boost` | `pklin` |
| `units` | `dimensionless` | `Mpc3` |
| `law` | `syren_halofit` | `syren_linear` |

The Syren law is an analytic matter-power calculation used as the starting
prediction. Its `train_base` and `val_base` files store that prediction for
the training and validation rows. Change both files to match the chosen
quantity. The full scientific explanation is in the main README's
[background](../README.md#16-emulating-the-expansion-history-hz-bao-and-sn-distances)
and [matter-power](../README.md#17-emulating-the-matter-power-spectrum-hybrid-inference-emul2)
appendices.

# Appendices about special runs <a id="appendix-c-special-runs"></a>

## FAQ C1. What is the difference between a sweep and tuning? <a id="faq-c1-sweep-tune"></a>

A sweep trains once for every value you list for one setting. The `sweep`
block is at the top level, beside `data` and `train_args`:

```yaml
sweep:
  parameter: lr.lr_base
  values:
    - 0.001
    - 0.0025
    - 0.0063
```

The parameter is the nested setting name below `train_args`. For example,
`lr.lr_base` means `train_args` → `lr` → `lr_base`. The sweep program
changes only that setting and holds the others fixed.

Tuning asks Optuna, a search package, to select several numeric settings. A
searched value has four entries: `[default, minimum, maximum, kind]`. `kind`
is `int`, `float`, or `log`.

```yaml
train_args:
  model:
    name: resmlp
    mlp:
      width: [128, 64, 256, int]
  lr:
    lr_base: [0.0025, 0.0001, 0.01, log]
```

The first number is the ordinary training value and the first tuning trial.
A normal training program uses that first number instead of searching.

Only cosmic shear has separate sweep and tuning templates in this folder.
For another family, copy its family file, add one `sweep` block or numeric
ranges, and run the matching `<family>_sweep_hyperparam_emulator.py` or
`<family>_tune_emulator.py` program. Replace `<family>` with `scalar`, `cmb`,
`baosn`, or `mps`. The main README explains
[one-setting sweeps](../README.md#a-one-knob-sweep) and
[multi-setting searches](../README.md#a-hyperparameter-search).

## FAQ C2. What is the difference between fine-tuning and transfer? <a id="faq-c2-reuse"></a>

Both start from a saved `.emul` and `.h5` pair.

Fine-tuning continues changing the numbers learned by the saved model. Its
source goes in `train_args.finetune.from`. Remove `train_args.model` because
the source supplies the model's structure.

Transfer keeps the saved base fixed and trains another model as a
correction. Its source goes in the top-level `transfer.from` setting. Keep
`train_args.model` because it describes the correction model. The supplied
example keeps the base fixed; its optional `refine` block is explained in the
main README.

The source and new run must describe compatible inputs and outputs. Read the
family checks and the full examples before starting either mode. The main
README lists the allowed combinations in
[How do I start from a saved emulator?](../README.md#13-starting-from-a-saved-emulator-fine-tuning--transfer).

## FAQ C3. How do I create the CMB covariance file? <a id="faq-c3-cmb-covariance"></a>

`cmb_covariance_lcdm.yaml` is input to a physics calculation. It fixes the
fiducial (reference) cosmology, CAMB calculation settings, experiment noise,
beam, sky fraction, and whether to calculate the much slower non-Gaussian
terms.

Run the following from `$ROOTDIR`. The calculation writes
`$ROOTDIR/projects/cmb/chains/cmbcov_lcdm.npz`. It can be expensive when
`cov_args.nongaussian.enabled` is `true`.

```bash
cd "$ROOTDIR"
D=external_modules/code/emulators_code_v2
PROJECT=projects/cmb
CONFIG_DIR=generator

mkdir -p "$PROJECT/$CONFIG_DIR"
cp "$D/example_yamls/cmb_covariance_lcdm.yaml" \
  "$PROJECT/$CONFIG_DIR/cmb_covariance_lcdm.yaml"

python "$D/compute_data_vectors/compute_cmb_covariance.py" \
  --root "$PROJECT" \
  --fileroot "$CONFIG_DIR" \
  --yaml cmb_covariance_lcdm.yaml \
  --output cmbcov_lcdm
```

Make the training YAML name that same result. With the standard CoCoA startup
links, keep the bare filename used by the training example:
`data.cmb.covariance: cmbcov_lcdm.npz`.

The data-generation guide explains the outputs and cost in
[Why is the CMB covariance a separate calculation?](../compute_data_vectors/README.md#faq-a5-cmb-covariance).

# Appendices about paths and errors <a id="appendix-d-paths"></a>

## FAQ D1. Where does each filename point? <a id="faq-d1-paths"></a>

The training programs treat four groups of paths differently:

| Settings | Where a relative name is read from |
| --- | --- |
| `train_dv`, `train_params`, `train_covmat`, `val_dv`, `val_params` | `$ROOTDIR/<project>/chains`, where `<project>` is the value passed to `--root`; for example, `$ROOTDIR/projects/lsst_y1/chains` |
| `cosmolike_data_dir` | A directory below `$ROOTDIR/external_modules/data` |
| `cosmolike_dataset` | A dataset below the selected `cosmolike_data_dir` |
| Supporting filenames inside `cmb`, `grid`, and `grid2d` | Exactly the path written in the YAML, relative to the folder where the program starts |

The documented commands start in `$ROOTDIR`, and the shipped YAMLs use bare
supporting filenames. `start_cocoa.sh` makes the corresponding files available
there through symbolic links; `stop_cocoa.sh` removes those links. A symbolic
link is a short pointer to a file stored elsewhere. The files remain in their
project `chains` folders, so no supporting data file is copied.

This applies only to supporting data. You should still copy the YAML you want
to edit into your project, as [Step 2](#copy-file) explains.

This rule currently applies to:

- `data.cmb.covariance`;
- `data.grid.z_file`;
- `data.grid2d.z_file` and `data.grid2d.k_file`; and
- `data.grid2d.train_base` and `data.grid2d.val_base`.

A bare supporting name is read as `$ROOTDIR/<name>`, normally through the link
created during startup. A project-relative path starts with `projects/...`;
an absolute path is the full path beginning with `/`. Both are read exactly as
written. Do not put the literal text `$ROOTDIR` in one of these YAML values
because these fields do not expand environment variables.

The five main input filenames in the first row of the table are different. A
bare name there is resolved directly under the selected project's `chains`
folder.

## FAQ D2. What should I check when a run stops at startup? <a id="faq-d2-startup-errors"></a>

| What you see | What it usually means | First action |
| --- | --- | --- |
| `YAML file not found` | `--fileroot` or `--yaml` points to another folder or name | Compare the command with the path created in step 2 |
| A YAML parser or scanner error | Indentation, punctuation, or a tab character broke the YAML syntax | Run the syntax check in step 4 and inspect the named line |
| `config did not parse to a mapping` | The file is empty, or its top level is a list or single value instead of named blocks | Compare its first active lines with the unchanged template |
| `unknown ... key` | A setting is misspelled, placed in the wrong block, or unsupported | Compare it with the unchanged template and the linked main README section |
| The error names another training program | The YAML's family block does not match the program | Use the program paired with that file in the chooser table |
| An input file is missing | A filename uses the wrong base folder or the data has not been generated | Apply FAQ D1, then check that the file exists |
| The startup summary names the wrong model or value | A command-line option overrides the YAML, or the same key appears twice | Remove the override or duplicate, then rerun the syntax and startup checks |

The example files do not create their inputs. To generate training and
validation rows, start with
[`compute_data_vectors/README.md`](../compute_data_vectors/README.md).
