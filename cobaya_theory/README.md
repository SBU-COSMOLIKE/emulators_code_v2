# Use saved emulators in Cobaya

Cobaya is a program that chooses cosmological parameter values, asks a
scientific calculation for predictions, and gives those predictions to a
likelihood. A likelihood compares a prediction with data and assigns it a
score.

This folder connects Cobaya to saved CoCoA SONIC emulators. An **emulator** is
a trained neural network that approximates a slower calculation. An
**adapter** is a small Python class that translates between Cobaya and one type
of saved emulator.

One saved emulator has two matching files: a `.h5` information file and a
`.emul` weights file. Both have the same path before the extension.

This folder does not train emulators or generate training data. Use
[`example_yamls/`](../example_yamls/README.md) to configure training and
[`compute_data_vectors/`](../compute_data_vectors/README.md) to make training
and validation tables.

CoCoA SONIC currently uses NumPy 1.x. Keep the supplied CoCoA environment; do
not upgrade it to NumPy 2.

```mermaid
flowchart LR
  C["Cobaya chooses parameters"] --> A["The adapter runs the configured saved emulators"]
  A --> P["The emulator returns a prediction"]
  P --> L["The likelihood scores the prediction"]
```

In words: Cobaya chooses a point, the adapter runs the matching saved files,
and the likelihood scores the returned physical prediction.

## Contents

### Main guide

1. [Prepare CoCoA](#prepare-cocoa)
2. [Choose an adapter and example](#choose-adapter)
3. [Copy the example](#copy-example)
4. [Edit your copy](#edit-copy)
5. [Check the setup](#check-setup)
6. [Run one evaluation](#run-evaluation)

### Common questions raised by developers

- [Appendices about saved emulator files](#appendix-a-files)
  - [Why does one saved emulator have two files?](#faq-a1)
  - [How are saved-emulator paths resolved?](#faq-a2)
  - [How many saved roots does each adapter accept?](#faq-a3)
  - [What does the adapter check when it loads files?](#faq-a4)
  - [Which device should I request?](#faq-a5)
  - [Why must I keep NumPy 1.x?](#faq-a6)
- [Appendices about the Cobaya YAML](#appendix-b-yaml)
  - [What do the main YAML blocks mean?](#faq-b1)
  - [How does Cobaya learn which parameters the network needs?](#faq-b2)
  - [What options may I place under extra_args?](#faq-b3)
  - [How do I write a scalar theory block?](#faq-b4)
  - [How do I write a cosmic-microwave-background theory block?](#faq-b5)
  - [How do I write a background-expansion theory block?](#faq-b6)
  - [How do I write a matter-power theory block?](#faq-b7)
  - [What is the difference between --test, evaluate, and MCMC?](#faq-b8)
  - [May I reuse an output name?](#faq-b9)
- [Appendices about each physical result](#appendix-c-results)
  - [What does emul_cosmic_shear return?](#faq-c1)
  - [How does emul_scalars combine named values?](#faq-c2)
  - [What does emul_cmb return?](#faq-c3)
  - [Why does emul_baosn need two redshift ranges?](#faq-c4)
  - [How does emul_mps form nonlinear matter power?](#faq-c5)
  - [What does the EMUL2 example replace?](#faq-c6)
- [Appendices about checks and errors](#appendix-d-checks)
  - [What does each check establish?](#faq-d1)
  - [What should I do when startup fails?](#faq-d2)
  - [What does a one-point failure leave on disk?](#faq-d3)
  - [When is the file ready for an MCMC?](#faq-d4)
  - [Can I call a saved emulator without Cobaya?](#faq-d5)

---

## 1. Prepare CoCoA <a id="prepare-cocoa"></a>

Install and start CoCoA by following the
[official CoCoA README](https://github.com/CosmoLike/cocoa/blob/main/README.md).
It explains how to activate the supplied environment and run
`start_cocoa.sh`.

That startup process defines `$ROOTDIR`, the path to your CoCoA folder. Run
all commands in this guide from `$ROOTDIR`. The supplied environment uses
NumPy 1.x; do not upgrade it to NumPy 2.

## 2. Choose an adapter and example <a id="choose-adapter"></a>

Choose the row that matches the quantity your likelihood needs.

| Quantity requested by Cobaya | Adapter | Saved emulators required |
| --- | --- | --- |
| Cosmic-shear data vector | `emul_cosmic_shear` | One data-vector emulator for the usual first run |
| Named values such as `H0` or `rdrag` | `emul_scalars` | One or more scalar emulators |
| Cosmic microwave background (CMB) TT, TE, EE, or lensing-potential spectrum | `emul_cmb` | One emulator for each requested spectrum |
| $H(z)$ and cosmological distances | `emul_baosn` | Exactly two: `Hubble` and `D_M` |
| Linear and nonlinear $P(k,z)$ | `emul_mps` | Exactly two: `pklin` and `boost` |

A **data vector** is the ordered list of predicted measurements that a
likelihood compares with its data. A **CMB spectrum** gives sky-fluctuation
power as a function of angular scale. The matter power spectrum $P(k,z)$
describes density fluctuations as a function of wavenumber $k$ and redshift
$z$.

This folder includes two templates with all the main Cobaya sections:

| Example | Use it when |
| --- | --- |
| [`EXAMPLE_EMUL_EVALUATE.yaml`](EXAMPLE_EMUL_EVALUATE.yaml) | You want the shortest first run with one cosmic-shear emulator |
| [`EXAMPLE_EMUL2_EVALUATE.yaml`](EXAMPLE_EMUL2_EVALUATE.yaml) | You already have five saved emulators for the advanced EMUL2 calculation |

They are not ready to run as downloaded. Supply the saved-emulator roots and
use each template inside a configured CoCoA project.

Both examples use Cobaya's `evaluate` sampler. Here, **evaluate** means
calculate the likelihood at one chosen parameter point. A Markov chain Monte
Carlo (MCMC) sampler instead evaluates many related points to estimate a
probability distribution.

EMUL2 is a CosmoLike mode in which five saved emulators provide several theory
quantities. CosmoLike is the likelihood code used by these LSST examples.

Start with `EXAMPLE_EMUL_EVALUATE.yaml` unless you specifically need EMUL2.
The appendices give theory blocks for the other adapters.

## 3. Copy the example <a id="copy-example"></a>

The template is already included at
`external_modules/code/emulators_code_v2/cobaya_theory/EXAMPLE_EMUL_EVALUATE.yaml`.
Using your editor or file manager, manually copy that file to a new name inside
the LSST Y1 project. This guide uses the following destination:

```text
projects/lsst_y1/my_emulator_evaluate.yaml
```

Choose a filename that is not already present. The LSST Y1 project keeps its
Cobaya YAML files directly in `projects/lsst_y1`; do not create a
`projects/lsst_y1/cobaya` subfolder. If you use another project or likelihood,
put the copy directly in that project and change the likelihood block too.

## 4. Edit your copy <a id="edit-copy"></a>

Open `projects/lsst_y1/my_emulator_evaluate.yaml` in a text editor.
Make these changes:

1. Replace `projects/lsst_y1/emulators/<run>/emul_v2` with your saved-emulator
   root.
2. Set `device: cpu` for the first check.
3. Change `output` to a new name that is not used by another run.
4. Check the likelihood `path`, `data_file`, and requested parameters.
5. Check every number under `sampler: evaluate: override:`.

A **saved-emulator root** is the shared path before the two file extensions.
For example, this YAML entry:

```yaml
emulators:
  - projects/lsst_y1/emulators/run_12/emul_v2
```

requires both files below:

```text
projects/lsst_y1/emulators/run_12/emul_v2.h5
projects/lsst_y1/emulators/run_12/emul_v2.emul
```

Do not add either extension to the YAML. Keep the two files from the same
training run together.

After editing, check that the shipped placeholder is gone:

```bash
CONFIG=projects/lsst_y1/my_emulator_evaluate.yaml
if ! test -f "$CONFIG"; then
  printf 'STOP: %s does not exist.\n' "$CONFIG" >&2
elif grep -n '<run>' "$CONFIG"; then
  printf '%s\n' 'STOP: replace <run> before continuing.'
else
  GREP_STATUS=$?
  if test "$GREP_STATUS" -eq 1; then
    printf '%s\n' 'No <run> placeholder remains.'
  else
    printf 'STOP: could not read %s.\n' "$CONFIG" >&2
  fi
fi
```

## 5. Check the setup <a id="check-setup"></a>

First check the YAML indentation and punctuation:

```bash
CONFIG=projects/lsst_y1/my_emulator_evaluate.yaml
python - "$CONFIG" <<'PY'
from pathlib import Path
import sys
from cobaya.input import load_input

path = Path(sys.argv[1])
document = load_input(str(path))
if not isinstance(document, dict):
    raise SystemExit("The YAML top level must contain named blocks.")
print(f"YAML syntax OK: {path}")
PY
```

This check reads Cobaya YAML syntax and refuses duplicate keys. It does not
open the saved emulator or decide whether the cosmology is correct.

Next ask Cobaya to load the likelihood, adapter, saved files, parameters, and
sampler without calculating the likelihood:

```bash
cd "$ROOTDIR"
CONFIG=projects/lsst_y1/my_emulator_evaluate.yaml
cobaya-run --test --no-mpi "$CONFIG"
```

`--test` initializes the model and sampler, then exits. `--no-mpi` tells
Cobaya not to start its MPI parallel-processing layer. A successful setup
ends with `Test initialization successful!`.

The setup check may write `*.input.yaml` and `*.updated.yaml` beside the
configured output prefix. Those files record the input and the defaults Cobaya
added. The setup check does not produce the one-point `*.1.txt` result.

## 6. Run one evaluation <a id="run-evaluation"></a>

Run the copied file without `--test`:

```bash
cd "$ROOTDIR"
CONFIG=projects/lsst_y1/my_emulator_evaluate.yaml
cobaya-run --no-mpi "$CONFIG"
```

This command performs a real likelihood calculation at the values under
`sampler: evaluate: override:`. It may take time because the likelihood still
runs even when an emulator replaces part of its theory calculation.

The terminal should report the reference point, log-prior, log-likelihood, and
log-posterior. The log-prior scores the point before comparing it with data;
the log-likelihood scores the data comparison; the log-posterior combines
both. Cobaya writes a one-row `<output>.1.txt` file plus the
`<output>.input.yaml` and `<output>.updated.yaml` records. The shipped example
sets `print_datavector: false`, so it does not write the named model-vector
file unless you change that setting.

One successful point proves that this parameter point, these saved files, and
this likelihood can run together. It does not prove emulator accuracy across
all allowed parameter ranges. Compare the emulator with **held-out validation
data**—examples that were not used for training—before starting an MCMC.

The [main CoCoA SONIC guide](../README.md#run-the-saved-emulator-in-a-cobaya-mcmc)
shows how to change a checked evaluate file into an MCMC file.

---

# Common questions raised by developers <a id="common-questions"></a>

The main guide is enough for the ordinary cosmic-shear check. Use the
appendices when you need another physical quantity, a path explanation, or a
specific error.

# Appendices about saved emulator files <a id="appendix-a-files"></a>

## FAQ A1. Why does one saved emulator have two files? <a id="faq-a1"></a>

The `.h5` file records the network recipe, parameter names, training ranges,
output description, and fixed scientific settings. The `.emul` file records
the learned network weights.

The adapter opens both files when Cobaya starts. Moving one file is allowed
only if you move its matching partner and keep the shared root. Never combine
the `.h5` file from one training run with the `.emul` file from another.

The code cannot always distinguish weights from another run when both networks
have the same shape. Keeping each `.h5` file beside its original `.emul`
partner is therefore part of preparing the run.

## FAQ A2. How are saved-emulator paths resolved? <a id="faq-a2"></a>

A **relative path** starts from another known folder. An **absolute path**
starts at the filesystem root, such as `/Users/name/...` on macOS or Linux.

Each adapter leaves an absolute saved-emulator root unchanged. When
`$ROOTDIR` is set, it joins a relative root to that folder, so this entry:

```yaml
emulators:
  - projects/lsst_y1/emulators/run_12/emul_v2
```

means:

```text
$ROOTDIR/projects/lsst_y1/emulators/run_12/emul_v2
```

Do not write a literal `$ROOTDIR` inside the YAML. YAML does not ask the shell
to expand that variable. Source `start_cocoa.sh` and run from `$ROOTDIR`
instead.

If `$ROOTDIR` is unset, a relative root starts from the folder where
`cobaya-run` was launched. This guide avoids that ambiguity by sourcing
`start_cocoa.sh` first.

`python_path` follows Cobaya's external-class rule. From `$ROOTDIR`, use:

```yaml
python_path: ./external_modules/code/emulators_code_v2/cobaya_theory/
```

Do not replace `python_path` with `path`. In a likelihood block, `path`
can name likelihood data; in these theory blocks, `python_path` selects the
adapter Python file.

## FAQ A3. How many saved roots does each adapter accept? <a id="faq-a3"></a>

| Adapter | Count | Additional rule |
| --- | ---: | --- |
| `emul_cosmic_shear` | One or more | Several vectors are joined in YAML order |
| `emul_scalars` | One or more | No two roots may return the same named value |
| `emul_cmb` | One or more | Use one root for each requested spectrum |
| `emul_baosn` | Exactly two | One `Hubble` root and one `D_M` root |
| `emul_mps` | Exactly two | One `pklin` root and one `boost` root |

The adapter reads the saved output description, so list order does not choose
the meaning of either exactly-two pair. Giving two roots with the same output
name is refused.

## FAQ A4. What does the adapter check when it loads files? <a id="faq-a4"></a>

The adapter checks that every root belongs to its physical family. When several
roots are used together, it also checks that they came from the same generated
dataset and agree on saved scientific settings and parameter coordinates.

After Cobaya has assembled the model, the adapter compares the saved fixed
settings with the active Cobaya settings. A **fixed setting** is a quantity
held constant rather than sampled. A mismatch stops startup.

At each prediction, every network input recorded in a saved file must be named
and inside that file's training range. Extra values requested by an adapter,
such as `fast_params` or the Syren inputs used by the matter-power adapter,
do not have a saved training range in that adapter.

Cobaya passes parameters by name, so their YAML order does not choose the
network order. Missing saved input names and saved inputs outside their
recorded ranges are refused.

These checks catch file and configuration disagreements. They do not measure
the emulator's validation accuracy.

## FAQ A5. Which device should I request? <a id="faq-a5"></a>

Use `device: cpu` for the first check. It runs the network on the computer's
central processor.

`device: cuda` requests an NVIDIA graphics processor (GPU). If CUDA is
unavailable, the adapter uses Apple's Metal Performance Shaders (MPS) device
when available, otherwise CPU. `device: mps` requests that Apple GPU interface
and falls back to CPU.

This fallback is silent, so the requested value is not evidence that a GPU
was selected. If GPU use matters, check PyTorch's CUDA or MPS availability
before starting the run.

`compile: false` is the default. `compile: true` asks PyTorch to compile the
model only when CUDA is selected and the saved recipe includes a compile mode.
Compilation startup time rarely helps one-point evaluations or MCMC calls that
predict one parameter point at a time, so leave it off unless you have measured
this run.

## FAQ A6. Why must I keep NumPy 1.x? <a id="faq-a6"></a>

CoCoA SONIC currently targets NumPy 1.x. The supplied environment contains the
version used by the repository's present calculations and tests.

Do not upgrade NumPy as part of a Cobaya adapter setup. If another package
changes the environment to NumPy 2, restore the CoCoA environment before
diagnosing the YAML or saved emulator.

# Appendices about the Cobaya YAML <a id="appendix-b-yaml"></a>

## FAQ B1. What do the main YAML blocks mean? <a id="faq-b1"></a>

| Block | Meaning |
| --- | --- |
| `likelihood` | The data comparison that requests a prediction |
| `params` | Sampled, fixed, and calculated parameter names |
| `theory` | The adapter classes that produce physical predictions |
| `sampler` | The rule for choosing parameter points |
| `output` | The path prefix for files written by Cobaya |

A **sampled parameter** is allowed to change during an MCMC. A **derived
parameter** is calculated from other parameters rather than sampled directly.

The shortest cosmic-shear theory block is:

```yaml
theory:
  emul_cosmic_shear:
    python_path: ./external_modules/code/emulators_code_v2/cobaya_theory/
    stop_at_error: true
    extra_args:
      device: cpu
      emulators:
        - projects/lsst_y1/emulators/run_12/emul_v2
```

The class name after `theory:` must match the adapter name in the chooser
table.

## FAQ B2. How does Cobaya learn which parameters the network needs? <a id="faq-b2"></a>

Each saved `.h5` file records the input parameter names in training order.
The adapter reads those names and asks Cobaya for the corresponding values.
Do not type an input order into the theory block.

Every saved input name must be available from `params`, the likelihood, or
another calculation in the Cobaya file. If a saved root expects `As_1e9`,
renaming or dropping that quantity before it reaches the adapter causes a
missing-parameter error.

The matter-power adapter needs extra quantities when it reconstructs a Syren
starting prediction. Syren supplies a matter-power formula that the emulator
corrects. The adapter asks for `As`, `ns`, `H0`, `omegab`, and `omegam`.
When the sampled amplitude is `As_1e9`, define `As` as a derived parameter,
as shown in [`EXAMPLE_EMUL2_EVALUATE.yaml`](EXAMPLE_EMUL2_EVALUATE.yaml).

## FAQ B3. What options may I place under `extra_args`? <a id="faq-b3"></a>

| Adapter | Allowed keys |
| --- | --- |
| `emul_cosmic_shear` | `device`, `emulators`, `fast_params`, `compile`, `dv_return` |
| `emul_scalars` | `device`, `emulators`, `provides`, `compile` |
| `emul_cmb` | `device`, `emulators`, `compile` |
| `emul_baosn` | `device`, `emulators`, `compile` |
| `emul_mps` | `device`, `emulators`, `compile` |

An unknown key stops initialization and prints the allowed list. Most runs need
only `device` and `emulators`.

## FAQ B4. How do I write a scalar theory block? <a id="faq-b4"></a>

```yaml
theory:
  emul_scalars:
    python_path: ./external_modules/code/emulators_code_v2/cobaya_theory/
    stop_at_error: true
    extra_args:
      device: cpu
      emulators:
        - projects/lsst_y1/emulators/rdrag/emul_v2
```

The saved file supplies its input names and output names. An optional
`provides` list only checks that named outputs exist; it does not select or
rename them.

## FAQ B5. How do I write a cosmic-microwave-background theory block? <a id="faq-b5"></a>

```yaml
theory:
  emul_cmb:
    python_path: ./external_modules/code/emulators_code_v2/cobaya_theory/
    stop_at_error: true
    extra_args:
      device: cpu
      emulators:
        - projects/cmb/emulators/tt/emul_v2
        - projects/cmb/emulators/te/emul_v2
        - projects/cmb/emulators/ee/emul_v2
```

Include only the spectra requested by the likelihood, and make sure each saved
range reaches the requested largest multipole.

## FAQ B6. How do I write a background-expansion theory block? <a id="faq-b6"></a>

```yaml
theory:
  emul_baosn:
    python_path: ./external_modules/code/emulators_code_v2/cobaya_theory/
    stop_at_error: true
    extra_args:
      device: cpu
      emulators:
        - projects/lsst_y1/emulators/baosn/hubble_v2
        - projects/lsst_y1/emulators/baosn/dm_v2
```

One saved emulator predicts low-redshift `Hubble`; the other predicts
high-redshift `D_M`. Their stored output descriptions, not their list order,
tell the adapter which is which.

## FAQ B7. How do I write a matter-power theory block? <a id="faq-b7"></a>

```yaml
theory:
  emul_mps:
    python_path: ./external_modules/code/emulators_code_v2/cobaya_theory/
    stop_at_error: true
    extra_args:
      device: cpu
      emulators:
        - projects/lsst_y1/emulators/mps/pklin_v2
        - projects/lsst_y1/emulators/mps/boost_v2
```

The two saved surfaces must have exactly the same stored $z$ and $k$ grids.
Syren supplies starting matter-power formulas that these emulators correct.
The intended EMUL2 pair uses the `syren_linear` formula for `pklin` and
`syren_halofit` for `boost`.

## FAQ B8. What is the difference between `--test`, evaluate, and MCMC? <a id="faq-b8"></a>

| Action | What it does | What it does not prove |
| --- | --- | --- |
| `cobaya-run --test` | Loads the complete setup and initializes the sampler | No likelihood value is calculated |
| `sampler: evaluate` | Calculates one or more chosen points | It does not explore the allowed parameter range |
| MCMC | Chooses many linked points to estimate a probability distribution | A completed chain alone does not validate emulator accuracy |

Run them in that order: setup-only check, one-point evaluation, validation
comparison, then MCMC.

## FAQ B9. May I reuse an output name? <a id="faq-b9"></a>

Use a new `output` prefix for a changed experiment. An old
`<output>.updated.yaml` tells Cobaya that the name has already been used.

Do not begin with `--force`; it authorizes Cobaya to overwrite prior output.
Do not delete an old chain merely to make a new YAML start. Give the new run a
new name so that both records remain available.

# Appendices about each physical result <a id="appendix-c-results"></a>

## FAQ C1. What does `emul_cosmic_shear` return? <a id="faq-c1"></a>

It returns the ordered cosmic-shear vector through Cobaya's
`get_cosmic_shear()` call. With one saved emulator, the result is that
emulator's vector. With several roots, the adapter joins the vectors in YAML
order.

`dv_return: section` is the default and returns only the vector positions for
the saved observable, called its **probe section**. `dv_return: 3x2pt`
returns the full 3x2pt-layout vector. This saved observable's kept entries
appear at their stored positions, and every other position is zero.

`fast_params` adds names that Cobaya must supply but does not feed those
values into the network or apply a correction. Use it only when another part
of the theory calculation handles those parameters.

## FAQ C2. How does `emul_scalars` combine named values? <a id="faq-c2"></a>

Each scalar saved emulator records one or more output names. The adapter returns
the combined set of those named values as Cobaya derived parameters.

Two loaded roots may not provide the same name. The output of one loaded scalar
emulator also may not be an input to another; chained scalar emulators are not
implemented.

## FAQ C3. What does `emul_cmb` return? <a id="faq-c3"></a>

It returns Cobaya's named `Cl` collection with a common integer multipole
$\ell$ axis. A **multipole** labels angular scale on the sky. Each saved
emulator contributes its stored TT, TE, EE, or PP values on its own stored
range.

The adapter returns raw $C_\ell$ values. It does not apply the common
$\ell(\ell+1)/(2\pi)$ plotting factor or convert units. TT, TE, and EE
files loaded together must record the same units; PP is dimensionless.

A likelihood cannot request a missing spectrum or a multipole above the
stored maximum.

## FAQ C4. Why does `emul_baosn` need two redshift ranges? <a id="faq-c4"></a>

The `Hubble` saved emulator covers the low-redshift supernova range and must
store $H(z)$ in km/s/Mpc. The adapter integrates this prediction to obtain
low-redshift distances.

The `D_M` saved emulator covers a separate high-redshift range near
recombination and must store distance in Mpc. The two stored ranges may not
overlap or touch. Requests in the gap or outside both ranges are refused.

`Hubble` queries work only in the low-redshift range. Distance queries may
use the low-redshift range or the stored high-redshift range.

Cobaya's setup-only check can accept a high-redshift `Hubble` request, but the
actual `get_Hubble()` call refuses it. Check that the likelihood requests
`Hubble` only inside the low-redshift range.

This adapter assumes a flat universe. Set `omk: 0` in the full Cobaya model.
The adapter rejects a sampled `omk` input, but a nonzero value fixed elsewhere
can escape that local check; the user must keep the complete run flat.

For two-redshift angular-diameter distances, supply each pair as
$(z_1,z_2)$ with $z_1 \le z_2$. The current calculation assumes this order
and does not check it.

## FAQ C5. How does `emul_mps` form nonlinear matter power? <a id="faq-c5"></a>

The `pklin` saved emulator gives the linear matter-power surface. The
`boost` saved emulator gives the multiplicative nonlinear correction
$B(k,z)$. The adapter forms:

$$
P_{\mathrm{nl}}(k,z) = B(k,z)P_{\mathrm{lin}}(k,z).
$$

For the intended EMUL2 files, Syren supplies starting matter-power formulas
and each network predicts a correction to one of them. The adapter reconstructs
both physical surfaces and applies its built-in low-$k$ blend when `boost`
uses `syren_halofit`. It serves only `delta_tot`–`delta_tot`, Cobaya's
total-matter-density spectrum.

The two saved roots must use identical stored $z$ and $k$ grids. A non-finite,
non-positive linear spectrum or boost rejects that parameter point.

The adapter checks the `pklin` and `boost` quantity names and exact grid
equality. It does not check units or prove that `pklin` uses
`syren_linear` while `boost` uses `syren_halofit`. For EMUL2, verify that
`pklin` stores `Mpc3` with `syren_linear` and that the dimensionless
`boost` uses `syren_halofit`. A `none` formula is also allowed when the
saved emulator learned the raw surface.

The current `sigma8` helper integrates with an 8 Mpc radius, not the usual
8 Mpc/$h$ radius used for the conventional $\sigma_8$ parameter. Obtain
conventional $\sigma_8$ from another checked calculation.

## FAQ C6. What does the EMUL2 example replace? <a id="faq-c6"></a>

EMUL2 is CosmoLike's `use_emulator: 2` mode. The CosmoLike likelihood still
runs, but five saved emulators replace the CAMB quantities it requests:

| Adapter | Saved roots | Returned quantity |
| --- | --- | --- |
| `emul_scalars` | One `rdrag` root | Sound horizon $r_\mathrm{drag}$ |
| `emul_baosn` | One `Hubble` and one `D_M` root | Expansion rate and distances |
| `emul_mps` | One `pklin` and one `boost` root | Linear and nonlinear matter power |

Five roots mean ten saved files because every root needs a matching `.h5`
and `.emul`.

CAMB is a program that calculates cosmological theory quantities. EMUL2 keeps
the likelihood calculation while replacing the listed CAMB work.

[`EXAMPLE_EMUL2_EVALUATE.yaml`](EXAMPLE_EMUL2_EVALUATE.yaml) also defines
calculated parameter names needed by the saved files and Syren formulas. Treat
it as an advanced integration check. Cobaya's setup-only check must accept the
whole copied file before you calculate the point.

The matter-power adapter reports that it can calculate power and `sigma8`,
but a particular likelihood may request parameters in a different form. The full
`cobaya-run --test` result, not the adapter's product list alone, decides
whether that YAML connects all requested quantities.

# Appendices about checks and errors <a id="appendix-d-checks"></a>

## FAQ D1. What does each check establish? <a id="faq-d1"></a>

| Check | A pass means | A pass does not mean |
| --- | --- | --- |
| YAML syntax check | Indentation, YAML punctuation, and keys can be read without duplicates | Paths and physics may still be wrong |
| `cobaya-run --test` | Components and saved files initialize together | The likelihood has not been calculated |
| One-point evaluate | One named point completes the likelihood | Other prior points and accuracy remain unchecked |
| Held-out validation | Errors are measured on rows not used for training | The Cobaya YAML and likelihood may still be wrong |
| MCMC | Cobaya explored many linked points | The emulator is scientifically accepted without the earlier checks |

Keep the setup-only log, one-point output, and validation plots with the run.
They answer different questions.

## FAQ D2. What should I do when startup fails? <a id="faq-d2"></a>

| Message or symptom | Likely meaning | First action |
| --- | --- | --- |
| A `.h5` or `.emul` file is missing | The YAML root is wrong or one partner was moved | Check both files beside the exact root |
| The saved emulator belongs in another adapter | The file predicts another physical family | Return to the chooser table |
| Saved settings do not match the Cobaya model | Files and current fixed cosmology disagree | Use files trained for this model or correct the YAML |
| A parameter is not provided | A saved input name cannot reach the adapter | Compare saved names with `params` and calculated names |
| A value lies outside the training range | The requested point exceeds saved limits | Correct the point or train for a wider range |
| Requested spectrum or multipole is unavailable | A cosmic-microwave-background file or stored $\ell$ range is missing | Add the correct spectrum or reduce the request |
| Requested redshift is unavailable | The point lies outside the background-expansion windows | Check both saved redshift ranges |
| Matter-power grids differ | The `pklin` and `boost` files use different axes | Use a pair generated on the same grid |
| The GPU appears unused | The requested device may have fallen back silently | Continue on CPU or check PyTorch's GPU availability |
| The output prefix already exists | Another check or run used the name | Choose a new output prefix |

Read the first adapter error before later Cobaya messages. A missing file or
parameter often causes several later messages that are only consequences.

## FAQ D3. What does a one-point failure leave on disk? <a id="faq-d3"></a>

Cobaya can write the `input` and `updated` YAML records before it evaluates
the point. A failed calculation may therefore leave those files without a
complete `.1.txt` row.

Do not read the presence of `updated.yaml` as proof of success. Check that the
command finishes without an error. Then open `.1.txt` and confirm that it
contains one data row.

For the next attempt, choose a new output prefix so the failed and repaired
configurations remain distinguishable.

## FAQ D4. When is the file ready for an MCMC? <a id="faq-d4"></a>

Move to an MCMC only after all of these are true:

- the YAML syntax check and `cobaya-run --test` pass;
- the one-point evaluation finishes with finite prior and likelihood values;
- held-out validation meets the accuracy requirement for the analysis;
- the saved training ranges contain the planned priors;
- units, parameter names, and fixed cosmology agree;
- background-expansion runs are flat and matter-power runs account for the
  `sigma8` definition;
- the MCMC uses a new output prefix.

The [main README](../README.md#run-the-saved-emulator-in-a-cobaya-mcmc)
contains the short MCMC conversion. Its appendices explain
[scalar outputs](../README.md#14-scalar-derived-parameter-emulators),
[CMB spectra](../README.md#15-emulating-cmb-spectra-tt--te--ee--phi-phi),
[background quantities](../README.md#16-emulating-the-expansion-history-hz-bao-and-sn-distances),
and [matter power](../README.md#17-emulating-the-matter-power-spectrum-hybrid-inference-emul2).

## FAQ D5. Can I call a saved emulator without Cobaya? <a id="faq-d5"></a>

Yes. Use `EmulatorPredictor` when a Python script needs a prediction but no
likelihood or Cobaya sampler. The
[direct-scripting appendix](../README.md#23-appendix-scripting-a-saved-emulator-without-cobaya)
gives the current Python interface and return shapes.

Use the adapters in this folder when Cobaya must decide parameter values,
connect predictions to a likelihood, and write sampling output.
