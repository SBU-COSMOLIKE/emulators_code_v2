# Generate training and validation data

This folder creates the numerical tables used to train and check an emulator.
An **emulator** is a neural network that learns to reproduce a slow physics
calculation much more quickly. These programs create its examples; they do
not train the neural network.

Each full generator run draws cosmological parameter values, calls CAMB or
CosmoLike once for each row, and saves the parameters and calculated
observables. **CAMB** calculates quantities such as the expansion history,
matter power, and CMB spectra for a cosmology. **CosmoLike** predicts survey
observables such as cosmic shear and galaxy clustering.

These calls can take far longer than an emulator training step. A production
run can also write many gigabytes of data.

Use an existing checked dataset when one is available. Run these programs
only when the scientific model, sampled region, output grid, and random seeds
have been chosen for a new dataset.

> **A generator command is not a preview.** It starts the physics calculations
> and writes files under `$ROOTDIR/<project>/chains/`. A fresh run can replace
> files that already use the same output names.

Here `<project>` means the folder supplied with `--root`. The worked example
below uses `projects/generator_example`.

```text
generator YAML + command-line choices
                  |
                  v
         sampled cosmologies
                  |
                  v
     CAMB or CosmoLike calculation
                  |
          +-------+--------+
          |                |
          v                v
  parameter table     observable arrays
          |                |
          +-------+--------+
                  |
                  v
       check every failure flag
                  |
                  v
       emulator trainer YAML
```

The YAML used here describes the physics calculation and the sampled
cosmologies. The later trainer YAML points to the saved tables and describes
the neural network. They are different files even though both contain a block
named `train_args`.

## Contents

### Main guide

1. [Prepare the CoCoA environment](#prepare-the-cocoa-environment)
2. [Choose the physics generator](#choose-the-physics-generator)
3. [Run the first 200-row calculation](#run-the-first-200-row-calculation)
4. [Check the saved rows](#check-the-saved-rows)
5. [Create a separate validation set](#create-a-separate-validation-set)
6. [Point the trainer at the files](#point-the-trainer-at-the-files)

### Common questions raised by developers

**[Appendices about generator inputs and outputs](#appendices-about-generator-inputs-and-outputs)**

- [FAQ A1. How is a generator YAML different from a trainer YAML?](#faq-a1-generator-yaml)
- [FAQ A2. What does each physics generator require and write?](#faq-a2-physics-families)
- [FAQ A3. How are the output filenames constructed?](#faq-a3-output-names)
- [FAQ A4. What is in the parameter and failure files?](#faq-a4-common-files)
- [FAQ A5. Why is the CMB covariance a separate calculation?](#faq-a5-cmb-covariance)

**[Appendices about sampling and computing](#appendices-about-sampling-and-computing)**

- [FAQ B1. What is the difference between uniform and tempered sampling?](#faq-b1-sampling-modes)
- [FAQ B2. What does each command-line option mean?](#faq-b2-command-options)
- [FAQ B3. Can I create only the parameter rows?](#faq-b3-chain-only)
- [FAQ B4. How do serial and MPI runs differ?](#faq-b4-mpi)
- [FAQ B5. How much memory and disk space will a run need?](#faq-b5-memory-disk)

**[Appendices about failures and interrupted runs](#appendices-about-failures-and-interrupted-runs)**

- [FAQ C1. What should I do when a row fails?](#faq-c1-failed-row)
- [FAQ C2. How do I resume a stopped run?](#faq-c2-resume)
- [FAQ C3. How do I add more rows?](#faq-c3-append)
- [FAQ C4. Which files in this folder are commands?](#faq-c4-program-files)

---

## Prepare the CoCoA environment

Follow the
[official CoCoA README](https://github.com/CosmoLike/cocoa/blob/main/README.md)
to install, compile, and start CoCoA. After completing those instructions,
`$ROOTDIR` is the top-level CoCoA folder and its Python environment is active.

Run the commands in this guide from `$ROOTDIR`. These two shortcuts name the
package folder and the active CoCoA Python program:

```bash
cd "$ROOTDIR"
D="$ROOTDIR/external_modules/code/emulators_code_v2"
PYTHON="$ROOTDIR/.local/bin/python"
```

The YAML files normally refer to CAMB and other CoCoA files with paths relative
to this folder.

Check the Python packages before starting a physics calculation. This command
does not create files:

```bash
"$PYTHON" -c "import cobaya, emcee, getdist, mpi4py, numpy, psutil, yaml; print('generator imports: OK')"
```

The expected final line is:

```text
generator imports: OK
```

The cosmic-shear, galaxy-lensing, and galaxy-clustering generator also needs
the compiled CosmoLike installation. The other three generators need the
CAMB installation named by their YAML.

Before every new run, write these choices in a lab notebook or a project note
outside `chains/`. For example, use
`$ROOTDIR/projects/generator_example/DATASET_NOTES.md`. Record:

- the generator YAML and the exact code version;
- the training or validation purpose;
- the number of rows;
- the output names;
- the random seed;
- whether sampling is uniform or tempered, and the fraction of each allowed
  parameter interval retained by `--boundary`;
- the expected time and disk use.

This command prints the code version as a Git commit ID. Paste its output into
the same note:

```bash
git -C "$D" rev-parse HEAD
```

Do not start a fresh run if its three output names belong to an existing
dataset that must be kept.

## Choose the physics generator

Choose one row. A **probe** is the observable family named by
`train_args.probe` in the generator YAML.

| Observable to calculate | Program | Accepted probe | Main output |
| --- | --- | --- | --- |
| Cosmic shear | `dataset_generator_lensing.py` | `cs` | one CosmoLike data-vector array |
| Galaxy-galaxy lensing | `dataset_generator_lensing.py` | `ggl` | one CosmoLike data-vector array |
| Galaxy clustering | `dataset_generator_lensing.py` | `gc` | one CosmoLike data-vector array |
| CMB spectra | `dataset_generator_cmb.py` | `cmblensed` or `cmbunlensed` | TT, TE, EE, and lensing-potential arrays |
| Expansion history | `dataset_generator_background.py` | `background` | $H(z)$ and radial comoving-distance $\chi(z)$ arrays |
| Matter power | `dataset_generator_mps.py` | `mps` | linear $P(k,z)$ and nonlinear-boost arrays |

One generator run can write several physical quantities. A later emulator
normally learns one of them. For example, the background generator writes
both $H(z)$ and $\chi(z)$, but the Hubble emulator reads only the $H(z)$ file.

The background program is used below because it requests background
quantities only. The example still performs 200 real CAMB calculations.

## Run the first 200-row calculation

This worked example varies $H_0$, the present expansion rate. It calculates
$H(z)$ on one eight-point redshift grid and radial comoving distance
$\chi(z)$ on a second eight-point grid. In a spatially flat cosmology,
$\chi(z)$ equals the transverse comoving distance $D_M(z)$; they differ when
spatial curvature is allowed to vary.

Run these commands from `$ROOTDIR` after completing the environment step.
The first command creates the folder that will hold the generator YAML:

```bash
mkdir -p "$ROOTDIR/projects/generator_example/generator"
```

Save the following file as
`$ROOTDIR/projects/generator_example/generator/background_minimal.yaml`:

```yaml
likelihood:
  background_anchor:
    external: "lambda _self: 0.0"
    requires:
      Hubble:
        z: [0.1]
        units: km/s/Mpc

theory:
  camb:
    path: ./external_modules/code/CAMB

params:
  H0:
    prior:
      min: 60.0
      max: 75.0
    latex: H_0
  ombh2:
    value: 0.02237
  omch2:
    value: 0.1200
  mnu:
    value: 0.06
  w:
    value: -1.0

train_args:
  probe: background
  ord:
    - [H0]
  z_sn: [0.0, 1.0, 8]
  z_rec: [1000.0, 1200.0, 8]
```

**Cobaya** is the program that joins parameter choices, physics calculations,
and comparisons with observations. It requires a `likelihood` block before it
will evaluate a theory. The example's `background_anchor` adds no
observational constraint: its small function always returns zero. Its
`requires` entry asks Cobaya for one Hubble value, which makes Cobaya call
CAMB.

The `theory` block selects CAMB. Its `path` is read relative to `$ROOTDIR`, so
the example uses the CAMB installation inside CoCoA.

The `params` block states which cosmological parameters vary and which stay
fixed. Here only $H_0$ varies; `prior.min` and `prior.max` are its allowed
limits. `latex` is only its display label.

The generator reads `train_args`. `probe: background` selects the background
calculation, while `ord` gives the saved column order for the varied
parameters. Each redshift entry is
`[lowest redshift, highest redshift, number of points]`.

Before creating `chains/` or any output file, the generator checks the full
YAML request. It checks the parameter names, the family-specific grids, and
the type and value of every setting in `train_args`. A malformed YAML
therefore stops before it can leave an empty output folder or a partial
dataset. [FAQ A1](#faq-a1-generator-yaml) gives the exact accepted forms.

Use new output names for this run:

```bash
cd "$ROOTDIR"

"$PYTHON" "$D/compute_data_vectors/dataset_generator_background.py" \
  --root projects/generator_example \
  --fileroot generator \
  --yaml background_minimal.yaml \
  --datavsfile dvs_train \
  --paramfile params_train \
  --failfile failed_train \
  --chain 0 \
  --nparams 200 \
  --unif 1 \
  --temp 1 \
  --seed 1234 \
  --freqchk 1000 \
  --loadchk 0 \
  --append 0
```

`--unif 1` draws directly from the finite $H_0$ interval. `--seed 1234`
fixes the random draws so the parameter table can be reproduced from the same
YAML and code. The programs require at least 200 rows. This command runs in
one process; [FAQ B4](#faq-b4-mpi) shows how to use more processes.

The command creates `projects/generator_example/chains/` and writes the
training files there. It may print progress and messages from CAMB. A zero
shell return code means the program reached its normal end; it does not prove
that every individual physics row succeeded. Check the failure file next.

## Check the saved rows

A failed physics calculation is stored as a row of zeros. The matching line
in the failure file is `1`. A successful row has a failure flag of `0`.

Run this check from `$ROOTDIR`. It reads the first example's files and changes
nothing:

```bash
"$PYTHON" - <<'PY'
from pathlib import Path
import numpy as np

folder = Path("projects/generator_example/chains")
prefix = "background_unifs"

params = np.atleast_2d(np.loadtxt(
    folder / f"params_train_{prefix}.1.txt"))
failures = np.atleast_1d(np.loadtxt(
    folder / f"failed_train_{prefix}.txt", dtype=np.uint8))
hubble = np.load(
    folder / f"dvs_train_{prefix}_h.npy", allow_pickle=False)
distance = np.load(
    folder / f"dvs_train_{prefix}_dm.npy", allow_pickle=False)
hubble_z = np.load(
    folder / f"dvs_train_{prefix}_h_z.npy", allow_pickle=False)
distance_z = np.load(
    folder / f"dvs_train_{prefix}_dm_z.npy", allow_pickle=False)

if params.shape[0] != failures.size:
    raise SystemExit("FAIL: parameter and failure row counts differ")
if hubble.shape != (failures.size, hubble_z.size):
    raise SystemExit("FAIL: H(z) array and redshift grid disagree")
if distance.shape != (failures.size, distance_z.size):
    raise SystemExit("FAIL: chi(z) array and redshift grid disagree")
if not set(np.unique(failures)).issubset({0, 1}):
    raise SystemExit("FAIL: failure file contains a value other than 0 or 1")
if failures.any():
    raise SystemExit(f"FAIL: {int(failures.sum())} physics rows failed")

print(
    f"PASS: {failures.size} parameter rows, "
    f"H {hubble.shape}, chi {distance.shape}, no failed rows"
)
PY
```

For this YAML, a fully successful result prints:

```text
PASS: 200 parameter rows, H (200, 8), chi (200, 8), no failed rows
```

Do not give an array to a trainer while its failure file contains `1`. Resume
the calculation as described in [FAQ C2](#faq-c2-resume), then rerun the
check.

## Create a separate validation set

The generators have no `--train` or `--validation` switch. Training and
validation are two different invocations.

Use three different output names and a different seed for validation. A
different filename alone does not create new cosmologies. The validation rows
must be absent from the training rows so that the validation score tests
predictions at unseen points.

```text
one reviewed generator YAML
          |
          +-- training:   names contain train, seed 1234,
          |               full allowed parameter intervals
          |
          +-- validation: names contain val, seed 5678,
                          central 90% of each allowed interval
```

The next command uses `--boundary 0.9`. The program removes five percent from
each end of every parameter interval, leaving the central 90 percent of each
interval. This value is an example for the worked dataset, not a universal
physics choice. Record the value chosen for a production study.

```bash
cd "$ROOTDIR"

"$PYTHON" "$D/compute_data_vectors/dataset_generator_background.py" \
  --root projects/generator_example \
  --fileroot generator \
  --yaml background_minimal.yaml \
  --datavsfile dvs_val \
  --paramfile params_val \
  --failfile failed_val \
  --chain 0 \
  --nparams 200 \
  --unif 1 \
  --temp 1 \
  --seed 5678 \
  --freqchk 1000 \
  --loadchk 0 \
  --append 0 \
  --boundary 0.9
```

The following check reads the validation files, checks their shapes and
failure flags, and confirms that no validation cosmology is also a training
cosmology. Run it from `$ROOTDIR` without editing it:

```bash
"$PYTHON" - <<'PY'
from pathlib import Path
import numpy as np

folder = Path("projects/generator_example/chains")
prefix = "background_unifs"
train_params = np.atleast_2d(np.loadtxt(
    folder / f"params_train_{prefix}.1.txt"))[:, 2:-1]
validation_params = np.atleast_2d(np.loadtxt(
    folder / f"params_val_{prefix}.1.txt"))[:, 2:-1]
failures = np.atleast_1d(np.loadtxt(
    folder / f"failed_val_{prefix}.txt", dtype=np.uint8))
hubble = np.load(
    folder / f"dvs_val_{prefix}_h.npy", allow_pickle=False)
distance = np.load(
    folder / f"dvs_val_{prefix}_dm.npy", allow_pickle=False)
hubble_z = np.load(
    folder / f"dvs_val_{prefix}_h_z.npy", allow_pickle=False)
distance_z = np.load(
    folder / f"dvs_val_{prefix}_dm_z.npy", allow_pickle=False)

if validation_params.shape[0] != failures.size:
    raise SystemExit("FAIL: parameter and failure row counts differ")
if hubble.shape != (failures.size, hubble_z.size):
    raise SystemExit("FAIL: H(z) array and redshift grid disagree")
if distance.shape != (failures.size, distance_z.size):
    raise SystemExit("FAIL: chi(z) array and redshift grid disagree")
if not set(np.unique(failures)).issubset({0, 1}):
    raise SystemExit("FAIL: failure file contains a value other than 0 or 1")
if failures.any():
    raise SystemExit(f"FAIL: {int(failures.sum())} physics rows failed")

training_rows = {tuple(row) for row in train_params}
overlap = sum(tuple(row) in training_rows for row in validation_params)
if overlap:
    raise SystemExit(
        f"FAIL: {overlap} validation rows also occur in the training set")
print(
    f"PASS: {failures.size} validation rows, valid shapes, "
    "no failures, and no training overlap"
)
PY
```

The first two columns in a parameter table are chain bookkeeping values. The
last column is `chi2*`. The check therefore compares only the cosmological
parameter columns between them.

For a production run, increase `--nparams` only after the 200-row calculation
has produced the expected file set, array shapes, physical variation, and
zero failure flags.

## Point the trainer at the files

The emulator trainer reads filenames from its own YAML. For the background
example, an $H(z)$ trainer would contain this `data` block:

```yaml
data:
  grid:
    quantity: Hubble
    units: km/s/Mpc
    law: log_offset
    offset: 0.0
    z_file: dvs_train_background_unifs_h_z.npy

  train_dv:     dvs_train_background_unifs_h.npy
  val_dv:       dvs_val_background_unifs_h.npy
  train_params: params_train_background_unifs.1.txt
  val_params:   params_val_background_unifs.1.txt
  train_covmat: params_train_background_unifs.covmat

  train_failure_mask: failed_train_background_unifs.txt
  val_failure_mask:   failed_val_background_unifs.txt
```

Every filename in the `data` block—the flat entries and the `z_file` name
inside the `grid` group—is read from `$ROOTDIR/<project>/chains/`, the folder
the generator wrote into. The two `failure_mask` entries name the generator's
failure files; training refuses to treat a failed row's zero-filled payload
as data, so a data-vector source must always name its mask.

The complete background trainer example is
[`example_yamls/baosn_hubble_emulator.yaml`](../example_yamls/baosn_hubble_emulator.yaml).
The CMB and matter-power trainer examples are
[`cmb_emulator.yaml`](../example_yamls/cmb_emulator.yaml) and
[`mps_boost_emulator.yaml`](../example_yamls/mps_boost_emulator.yaml).

Those three files configure emulator training. Do not pass them to a program
in this folder as generator YAMLs.

After generating and checking the tables, follow the main README to
[train the saved tables](../README.md#start-run). The appendices below are the
detailed reference for generating those tables.

---

# Common questions raised by developers

# Appendices about generator inputs and outputs <a id="appendices-about-generator-inputs-and-outputs"></a>

## FAQ A1. How is a generator YAML different from a trainer YAML? <a id="faq-a1-generator-yaml"></a>

A generator YAML tells Cobaya which physics program to call, which parameters
to vary, and which observable grid to save. Cobaya is the program that joins
those parameters, the physics calculation, and any comparison with
observations. The generator itself requires these top-level blocks:

| Block | Meaning here |
| --- | --- |
| `params` | varied and fixed cosmological parameters |
| `likelihood` | a comparison with observations, or a small function that returns zero when no observational constraint is wanted but a theory calculation must still run |
| `train_args` | the probe, parameter order, output grid, and allowed sampling intervals used by the generator |

The supplied physics configurations normally also include a `theory` block.
For the background, matter-power, and CMB families, that block selects and
configures CAMB. A CosmoLike configuration may obtain its theory through its
configured components instead.

Every generator needs `train_args.probe` and `train_args.ord`. `ord` has one
outer list containing one ordered list of varied parameter names:

```yaml
train_args:
  probe: background
  ord:
    - [H0, ombh2, omch2]
```

The names must be nonempty, unique single tokens with no spaces, line breaks,
or control characters, and identical to Cobaya's varied parameters. Their
order becomes the order of the saved parameter columns.
The extra outer list is required. For example, neither `ord: [H0, ombh2]`
nor two inner lists are accepted.

With `--unif 0`, `train_args` also needs `fiducial`, a mapping from every
varied name to its reference value, and `params_covmat_file`, the parameter
covariance file beside the YAML:

```yaml
train_args:
  probe: background
  ord:
    - [H0, ombh2]
  fiducial:
    H0: 67.4
    ombh2: 0.0224
  params_covmat_file: parameter_covariance.txt
```

With `--unif 1`, omit `fiducial` and `params_covmat_file`. They belong only
to the tempered MCMC request. Each sampling mode accepts its exact set of
`train_args` fields so that a stale or misspelled setting cannot be ignored.

The covariance text file begins with one header such as:

```text
# H0 ombh2 omch2
```

The filename must name one file directly beside the YAML; an absolute path,
parent path such as `../cov.txt`, or nested path is refused. The remaining
lines form one finite, square matrix. Opposite entries must agree up to tiny
floating-point roundoff. The generator then averages those roundoff-level
differences so the accepted matrix is exactly symmetric. A larger disagreement
is refused.

The number of header names must equal the number of matrix rows and columns.
Every diagonal entry is a variance and must be positive. The file may contain
parameters that are not varied in this run, but every name in `ord` must occur
exactly once. The generator checks this full header-to-matrix alignment before
selecting the smaller covariance used by the requested parameters. That
selected matrix must also produce a finite covariance factor and finite
inverse; a matrix that cannot support those calculations is refused.

YAML types are checked without converting text or Booleans into numbers.
Write a point count as `8`, not `8.0`, `"8"`, or `true`. Write a switch as
`false`, not `0` or `"false"`. Grid limits, extrapolation limits, and
fiducial values must be finite YAML numbers rather than quoted text. The
matter-power `write_syren_base` switch must be a real YAML Boolean; the
quoted strings `"true"` and `"false"` are both rejected.

Each family driver validates its own `train_args` grids and limits (ranges,
ordering, and minimum sizes) before the generator creates output. If tempered
MCMC sampling later produces fewer unique rows than requested, the run warns
and continues with the smaller table; the row count is visible in every
saved file.

The optional `latex` entry under a Cobaya parameter only controls its display
label. If `latex` is absent, `null`, or blank, the saved GetDist label is the
parameter name. Missing display text does not invalidate a scientific
parameter definition. A readable label may contain spaces and LaTeX commands,
but line breaks, tabs, NUL bytes, and other control characters are refused
before the sidecar is written.

A trainer YAML describes the saved arrays, neural network, loss, optimizer,
and training schedule. Its top-level `data` block points to the files created
here. The trainer's `train_args` block controls neural-network training; it is
not read by these generators.

## FAQ A2. What does each physics generator require and write? <a id="faq-a2-physics-families"></a>

### CosmoLike data vectors

`dataset_generator_lensing.py` accepts `cs`, `ggl`, or `gc`. The first
likelihood in its generator YAML must supply CosmoLike's `get_datavector`
method.

It writes one array:

```text
<data-name>_<probe>_<sampling-tag>.npy
```

The array shape is `(number of cosmologies, data-vector length)`. Each row is
one flat CosmoLike vector. This path needs a compiled CoCoA and CosmoLike
installation.

### CMB spectra

`dataset_generator_cmb.py` accepts `cmblensed` or `cmbunlensed`. Its
`train_args` block adds:

```yaml
train_args:
  probe: cmblensed
  ord:
    - [As, ns, H0]
  lrange: [2, 5000]
```

`lrange` includes both endpoints and must satisfy
`2 <= lowest multipole < highest multipole`. One CAMB call produces four
arrays:

Both multipoles must be ordinary YAML integers. Values such as `2.0`, `"2"`,
or `true` are not accepted in place of `2`.

```text
<data-name>_tt.npy
<data-name>_te.npy
<data-name>_ee.npy
<data-name>_pp.npy
```

Each shape is `(number of cosmologies, highest-lowest+1)`. TT, TE, and EE are
raw $C_\ell$ in $\mu\mathrm{K}^2$. PP is the raw dimensionless
lensing-potential spectrum.

The program does not write a separate multipole array; the current coordinate
record is `train_args.lrange`. The CMB covariance used by the trainer must
cover the same multipoles.

One CMB emulator learns one spectrum file. It does not combine all four files
into one target.

### Background expansion

`dataset_generator_background.py` requires:

```yaml
train_args:
  probe: background
  ord:
    - [H0]
  z_sn: [0.0, 3.0, 120]
  z_rec: [1000.0, 1200.0, 24]
```

The low-redshift grid starts at zero. The recombination grid has positive,
increasing limits. Each grid needs at least eight points, and the low-redshift
grid must end below the beginning of the recombination grid.
The two limits are finite YAML numbers. The point count is an ordinary YAML
integer, not a decimal, quoted value, or Boolean.

The program writes:

```text
<data-name>_h.npy       H(z), in km/s/Mpc
<data-name>_h_z.npy     redshifts for H(z)
<data-name>_dm.npy      radial comoving distance chi(z), in Mpc
<data-name>_dm_z.npy    redshifts for chi(z)
```

One emulator learns either the Hubble array or the distance array.

### Matter power

`dataset_generator_mps.py` requires:

```yaml
train_args:
  probe: mps
  ord:
    - [As, ns, H0]
  z_segments:
    - [0.0, 2.0, 16, false]
  k_log10: [-4.0, 1.0, 40]
  extrap_kmax: 10.0
  write_syren_base: false
```

Each redshift entry is
`[lowest z, highest z, number of points, include highest endpoint]`. The
combined grid must increase without repeated values and must contain at least
four points. `k_log10` defines a logarithmic wavenumber grid with at least
eight points. `extrap_kmax` must reach the highest requested $k$.

Grid limits and `extrap_kmax` are finite YAML numbers. Point counts are
ordinary YAML integers. The endpoint switch is the unquoted YAML Boolean
`true` or `false`.

Use the YAML booleans `true` or `false` without quotation marks for
`write_syren_base`.

The program always writes:

```text
<data-name>_pklin.npy   linear P(k,z), in Mpc^3
<data-name>_boost.npy   P_nonlinear/P_linear, dimensionless
<data-name>_z.npy       redshift coordinates
<data-name>_k.npy       wavenumber coordinates
```

With `write_syren_base: true`, it also writes `pklin_base` and `boost_base`
arrays. Every surface row is flattened in redshift-then-wavenumber order. Its
length is `number of redshifts * number of wavenumbers`.

### Dark-energy coordinates in matter-power data

`w` and `w0` are two names for the present-day dark-energy value. `wa`
describes how that value changes with cosmic time. `w0pwa` is the sum
`w + wa`.

The matter-power generator accepts two equivalent ways to describe a
time-varying dark-energy point:

```text
w or w0, together with wa
w or w0, together with w0pwa = w + wa
```

For example, `w = -0.9` and `w0pwa = -0.7` mean `wa = 0.2`. The generator
records `dark_energy_law: w0wa-cpl` and
`dark_energy_inputs: [w, wa]` in the dataset description. The sampled row
still comes from Cobaya.

With `write_syren_base: true`, the program compares every repeated form before
calculating the two Syren starting surfaces. It stops if `w` and `w0`
disagree, if `w0pwa` does not equal `w + wa`, or if a time-varying model does
not provide enough information to determine both values. A missing `wa`
becomes zero only when the resolved run has been classified and saved as a
constant-`w` or cosmological-constant model.

## FAQ A3. How are the output filenames constructed? <a id="faq-a3-output-names"></a>

All outputs go under:

```text
$ROOTDIR/<--root>/chains/
```

`--datavsfile`, `--paramfile`, and `--failfile` are names, not output
directories. The program removes any parent folder and extension supplied in
those values. For example, `--datavsfile trial.npy` becomes the name `trial`.

Uniform sampling, `--unif 1`, adds:

```text
_<probe>_unifs
```

Tempered sampling, `--unif 0`, adds:

```text
_<probe>_<temperature>
```

The worked training command therefore turns `params_train` into
`params_train_background_unifs` before adding `.1.txt`, `.covmat`, and the
other parameter-file extensions.

A fresh run refuses when a completed dataset already owns the same resolved
name. Use names that identify the scientific dataset and whether it is
training or validation data.

## FAQ A4. What is in the parameter and failure files? <a id="faq-a4-common-files"></a>

Every full generator run writes these files in addition to its physical
arrays:

| File ending | Contents |
| --- | --- |
| `.1.txt` | chain weight, saved log probability (`lnp`), sampled parameters in `ord` order, and `chi2*`, which is `-2 * lnp` |
| `.paramnames` | the declared name and display label for each numeric parameter column |
| `.ranges` | the sampled lower and upper bounds |
| `.covmat` | covariance of the saved parameter rows |
| `.facts.yaml` | generator family, fixed cosmology, parameter order, requested and used bounds, and a digest of the parameter table |
| failure `.txt` | one `0` or `1` for each physical row; `1` means the calculation failed |

On a fresh run, the first comment line in `.1.txt` records the sampling seed
and random-number generator. An append run draws from a stream derived from
that seed together with the number of rows already saved, so it never repeats
the original rows and remains reproducible from the recorded inputs. A
uniform run stores `1` as a placeholder
`lnp`; a tempered run stores emcee's log probability. The asterisk marks
`chi2*` as a derived GetDist column rather than a sampled parameter.

Physical arrays use 32-bit floating-point numbers. The parameter table is
decimal text written from the sampled values. Keep the parameter files,
physical arrays, coordinate arrays, and failure file together. They describe
one dataset.

## FAQ A5. Why is the CMB covariance a separate calculation? <a id="faq-a5-cmb-covariance"></a>

The CMB spectrum generator varies cosmologies and writes TT, TE, EE, and PP
rows. The CMB covariance program instead evaluates one fixed flat
$\Lambda$CDM cosmology for a chosen experiment. Its noise, beam, and sky
fraction belong to the experiment rather than to each training row.

Copy the shipped configuration into the project before editing it. These
commands create the project folder and one YAML file:

```bash
mkdir -p "$ROOTDIR/projects/cmb/generator"
cp "$D/example_yamls/cmb_covariance_lcdm.yaml" \
  "$ROOTDIR/projects/cmb/generator/cmb_covariance_lcdm.yaml"
```

Review the fixed cosmology, `lmax`, noise, beam, and `fsky`. Then run the
serial covariance program from `$ROOTDIR`:

```bash
cd "$ROOTDIR"

"$PYTHON" "$D/compute_data_vectors/compute_cmb_covariance.py" \
  --root projects/cmb \
  --fileroot generator \
  --yaml cmb_covariance_lcdm.yaml \
  --output cmb_covariance
```

The requested output name must be unused. Before the program reads the YAML or
starts CAMB, it checks for:

```text
$ROOTDIR/projects/cmb/chains/cmb_covariance.npz
```

If that path already exists, the command stops without changing it. Choose a
fresh value for `--output`, or move the earlier file explicitly before running
the calculation again. The program writes under a private temporary name and
gives it the public name only after `numpy.savez` finishes. That final step
refuses to replace a file created by another process.

Gaussian mode performs one fiducial CAMB calculation. It writes the
multipole coordinates, per-spectrum standard deviations, Gaussian
cross-spectrum terms, fiducial spectra, and a JSON provenance record inside
the `.npz` file. Here **fiducial** means the fixed reference cosmology, and
**provenance** means the saved record of how the file was made.

### Which Gaussian covariance does the file contain?

The Gaussian calculation uses raw angular power spectra, denoted by
$C_\ell^{XY}$. Here $X$ and $Y$ are either temperature $T$ or E-mode
polarization $E$. TT, TE, and EE spectra and their noise powers are in
$\mu\mathrm{K}^2$, where $\mu\mathrm{K}$ means microkelvin. The PP spectrum,
$C_\ell^{PP}=C_\ell^{\phi\phi}$, is the dimensionless lensing-potential
spectrum.

The experiment YAML gives a map-noise amplitude $\Delta_{XY}$ in
$\mu\mathrm{K}$-arcmin and a beam full width at half maximum $b$ in
arcminutes. Let

$$
q={\pi\over 180\times60}
$$

convert arcminutes to radians. The program calculates the instrumental noise
power

$$
N_\ell^{XY}=(q\Delta_{XY})^2
\exp\!\left[{\ell(\ell+1)(qb)^2\over8\ln 2}\right]
$$

for TT, TE, and EE. The configured names are `delta_tt`, `delta_te`,
`delta_ee`, and `beam_fwhm`. A zero `delta_te` means that the temperature and
polarization map noise is uncorrelated. The program also requires
$\Delta_{TE}^2\leq\Delta_{TT}\Delta_{EE}$; otherwise the two-field noise
covariance is not physically valid.

Define the signal-plus-noise spectra and the number of observed modes as

$$
\overline C_\ell^{XY}=C_\ell^{XY}+N_\ell^{XY},
\qquad
d_\ell=(2\ell+1)f_{\rm sky}.
$$

The symbols in these equations mean:

| Symbol | Meaning |
| --- | --- |
| $\ell$ | integer angular multipole, from 2 through the configured `lmax` |
| $X,Y,W,Z$ | field labels: temperature $T$ or E-mode polarization $E$ |
| $q$ | conversion from arcminutes to radians, $\pi/(180\times60)$ |
| $\Delta_{XY}$ | configured map-noise amplitude for fields $X,Y$, in $\mu\mathrm{K}$-arcmin |
| $b$ | configured beam full width at half maximum, in arcminutes |
| $\ln 2$ | natural logarithm of 2, used in the Gaussian beam factor |
| $C_\ell^{XY}$ | fiducial signal spectrum returned by CAMB for fields $X$ and $Y$ |
| $N_\ell^{XY}$ | beam-amplified instrumental noise power calculated above |
| $\overline C_\ell^{XY}$ | signal plus noise, $C_\ell^{XY}+N_\ell^{XY}$ |
| $f_{\rm sky}$ | fraction of the sky observed by the experiment; the YAML requires $0<f_{\rm sky}\leq1$ |
| $d_\ell$ | approximate number of measured modes at multipole $\ell$ after the sky-fraction correction |
| $V_\ell^{XY}$ | Gaussian variance of the named spectrum at one multipole |
| $\sigma_\ell^{XY}$ | positive error scale saved in the `.npz`, equal to $\sqrt{V_\ell^{XY}}$ |
| $G_\ell^{XY,WZ}$ | saved Gaussian covariance between two different spectra at the same multipole |

The four per-spectrum variances are

$$
\begin{aligned}
V_\ell^{TT} &={2(\overline C_\ell^{TT})^2\over d_\ell},\\
V_\ell^{TE} &={\overline C_\ell^{TT}\overline C_\ell^{EE}
                    +(\overline C_\ell^{TE})^2\over d_\ell},\\
V_\ell^{EE} &={2(\overline C_\ell^{EE})^2\over d_\ell},\\
V_\ell^{PP} &={2(C_\ell^{PP})^2\over d_\ell}.
\end{aligned}
$$

Gaussian mode also saves the same-multipole cross-spectrum blocks

$$
\begin{aligned}
G_\ell^{TT,TE} &={2\overline C_\ell^{TT}\overline C_\ell^{TE}\over d_\ell},\\
G_\ell^{TT,EE} &={2(\overline C_\ell^{TE})^2\over d_\ell},\\
G_\ell^{TE,EE} &={2\overline C_\ell^{EE}\overline C_\ell^{TE}\over d_\ell}.
\end{aligned}
$$

These arrays are saved as `gauss_tt_te`, `gauss_tt_ee`, and
`gauss_te_ee`. Gaussian mode treats different multipoles as uncorrelated. The
optional non-Gaussian calculation described below adds correlations between
multipoles.

The PP variance above follows the saved-file policy that the program calls
V1. It contains cosmic variance but omits lensing reconstruction noise
$N_\ell^{(0)}$, the noise from reconstructing the lensing potential from CMB
maps. The provenance record saves this omission under `pp_noise_n0`.

This equation is not a general PP covariance formula. Adding reconstruction
noise requires a new declared policy and a newly generated covariance file.

All of these equations use one fixed flat-$\Lambda$CDM fiducial cosmology and
the sky-fraction mode-counting approximation. They use raw $C_\ell$, not the
$\ell(\ell+1)C_\ell/(2\pi)$ plotting convention. The trainer reads the four
saved `sigma_*` arrays as its diagonal error scales.

`cov_args.nongaussian.enabled: true` adds six dense covariance blocks. Its
re-lensing count is:

```text
number of lensing bands * number of step sizes * 4
```

The six saved blocks alone need about
`6 * number_of_multipoles^2 * 8` bytes. The calculation also builds larger
temporary matrices. Check the requested memory and time before enabling this
mode. The current CMB trainer reads the diagonal scales and fiducial spectra;
it does not read the optional dense blocks.

The spectrum generator's `lrange` and the covariance file's `ell` array must
describe the same multipoles.

---

# Appendices about sampling and computing <a id="appendices-about-sampling-and-computing"></a>

## FAQ B1. What is the difference between uniform and tempered sampling? <a id="faq-b1-sampling-modes"></a>

### Why should the sampled region extend past the posterior?

The **posterior** is the region of parameter space favored after combining a
prior with the data likelihood. The **training support** is the region covered
by the cosmologies supplied to the emulator. A later inference run should not
need the emulator to predict beyond that training support.

[![Conceptual two-parameter picture with a posterior-like ellipse inside a broader training region](../documentation/figures/fig03_training_cloud.png)](../documentation/figures/fig03_training_cloud.pdf)

The two axes stand for any two sampled cosmological parameters. The blue
ellipse is a posterior-like region. The wider orange region is the intended
training coverage. This is a teaching picture, not a contour measured from a
saved dataset.

For tempered sampling, `--temp` broadens the covariance used by the linked
random walks in emcee, the sampling program described below. `--maxcorr`
limits the magnitude of its correlations, which prevents the sampled region
from remaining extremely thin across a strong parameter degeneracy. A
**degeneracy** is a direction in which several parameter combinations give
similar likelihood values.

Validation rows should remain inside the training support. A valid
`--boundary` value below one contracts each allowed interval before those
rows are drawn. Uniform sampling follows the final allowed intervals rather
than the ellipses in this picture.

`--unif 1` draws each parameter inside allowed intervals derived from the
Cobaya prior. For a finite prior, each interval begins with the stated lower
and upper bounds. The generator replaces infinite endpoints using `--temp`,
applies `--boundary` when requested, and moves off the exact endpoints by one
floating-point step.

Cobaya's raw confidence-one prior can report an infinite endpoint for an
unbounded parameter. That raw value is allowed only as an input to this
interval calculation. The resolved bounds used for sampling and written to
the parameter files must be finite and increasing. A finite raw endpoint that
is too large for the generator's `float32` parameter storage is refused; it is
not reinterpreted as an intentionally open endpoint.

Uniform mode is the better first run because it does not start an emcee chain.
The output name ends in `_unifs`.

`--unif 0` uses **emcee**, a Python program that explores nearby parameter
combinations with many linked random walks. Those walks begin around a
**fiducial cosmology**, the reference parameter values recorded in the YAML.
Their scale comes from the covariance file named by
`train_args.params_covmat_file`.

Temperature widens the explored region, and `--maxcorr` limits correlations
in that covariance. The output name ends in the numeric temperature.

A small requested row count does not make `--unif 0` a small trial. The code
prepares at least five million emcee candidate rows before selecting the
requested rows. Use uniform sampling for the first end-to-end calculation.

`--boundary VALUE` accepts `0 < VALUE <= 1`. A value below one contracts each
allowed parameter interval to the stated central fraction. For example,
`0.9` keeps the central 90 percent. Omission or `1.0` keeps the entire
interval. Zero, a negative value, a value above one, or a nonfinite value is
refused rather than silently replaced.

## FAQ B2. What does each command-line option mean? <a id="faq-b2-command-options"></a>

The four dataset generators share this command interface. Required means the
argument parser refuses to start without the option.

| Option | Required | Meaning and accepted value |
| --- | --- | --- |
| `--root` | yes | project folder below `$ROOTDIR` |
| `--fileroot` | yes | folder below the project that contains the generator YAML and any parameter covariance file |
| `--yaml` | yes | filename below `--fileroot`; this interface does not accept an arbitrary absolute YAML path |
| `--datavsfile` | yes | base name for the physical arrays |
| `--paramfile` | yes | base name for the parameter files |
| `--failfile` | yes | base name for the row-failure file |
| `--chain` | no | `0` for a full dataset; `1` for parameter files only; default `0` |
| `--nparams` | yes | requested total rows for a fresh run, or requested added rows for append; required but ignored for resume; integer at least 200 |
| `--unif` | yes | `1` to draw independent points across the resolved allowed intervals; `0` to use the linked emcee walks described in FAQ B1 |
| `--temp` | no | integer at least 1; default 128; always pass it so the chosen sampling intervals can be reconstructed from the command |
| `--maxcorr` | no | tempered-mode correlation limit; default 0.15; must satisfy `0.01 < value <= 1` |
| `--loadchk` | no | `1` to load the existing complete file set; default `0` |
| `--freqchk` | no | rows between intermediate saves; default 5000 and minimum 1000 |
| `--append` | no | `1` to add new rows after loading; default `0`; requires `--loadchk 1` |
| `--boundary` | no | remaining fraction of each parameter interval; finite value in `(0, 1]`; omission or `1.0` keeps the entire interval |
| `--seed` | yes | non-negative integer used for random draws in a fresh or append invocation |

Unknown or misspelled options are refused. The three legal run-control pairs
are:

```text
--loadchk 0 --append 0   start a fresh result set
--loadchk 1 --append 0   load it and retry failed rows
--loadchk 1 --append 1   load it and add new rows
```

## FAQ B3. Can I create only the parameter rows? <a id="faq-b3-chain-only"></a>

Yes. `--chain 1` samples parameters but skips every CAMB or CosmoLike target
calculation. It adds `_chain_only` to the names and writes only:

```text
.1.txt
.paramnames
.ranges
.covmat
.facts.yaml
```

It does not create or reuse a failure file, physical array, or coordinate
array. A chain-only result is not training data.

## FAQ B4. How do serial and MPI runs differ? <a id="faq-b4-mpi"></a>

MPI starts several Python processes for one command. One process is called a
**rank**. Rank 0 samples the cosmologies, calculates the first row to learn the
output shape, sends the remaining rows to workers, and writes the files.

The worked command starts one rank and calculates every row in that process.
To run four ranks, place `mpirun -n 4` before the same command:

```bash
mpirun -n 4 "$PYTHON" \
  "$D/compute_data_vectors/dataset_generator_background.py" \
  --root projects/generator_example \
  --fileroot generator \
  --yaml background_minimal.yaml \
  --datavsfile dvs_mpi_trial \
  --paramfile params_mpi_trial \
  --failfile failed_mpi_trial \
  --chain 0 \
  --nparams 200 \
  --unif 1 \
  --temp 1 \
  --seed 9012 \
  --freqchk 1000 \
  --loadchk 0 \
  --append 0
```

This is another real 200-row calculation and writes a new dataset. Do not run
it merely to inspect the command form.

In an MPI run, one row may use at most 1,800 seconds before rank 0 marks the
active work failed, saves a checkpoint, and stops the MPI job. Workers have
300 seconds to confirm shutdown. The serial path has no matching per-row
timer. `compute_cmb_covariance.py` is serial and should not be placed under
`mpirun`.

## FAQ B5. How much memory and disk space will a run need? <a id="faq-b5-memory-disk"></a>

Every physical target uses 32-bit floating-point values, or four bytes per
number. A first storage estimate is:

```text
number of rows * numbers per row * 4 bytes * number of target arrays
```

For CMB, there are four target arrays. For matter power, there are two target
arrays, or four when both Syren base arrays are requested. A matter-power row
contains `number of redshifts * number of wavenumbers` values in each array.

The program keeps target arrays in memory only when its estimate is below 75
percent of the currently available memory. Otherwise it uses disk-backed
NumPy arrays and prints that disk access will be slower. Appending may need a
temporary copy while an array grows, so free disk should exceed the final
dataset estimate.

The estimate does not include CAMB, CosmoLike, MPI-process memory, parameter
files, checkpoints, or temporary covariance matrices. Measure the 200-row run
before setting the production row count.

---

# Appendices about failures and interrupted runs <a id="appendices-about-failures-and-interrupted-runs"></a>

## FAQ C1. What should I do when a row fails? <a id="faq-c1-failed-row"></a>

Most per-row physics errors do not stop the complete run. The program writes
zeros for that target row, writes `1` at the same position in the failure
file, and continues. A physical target can also contain real zeros, so the
target array alone cannot identify a failed row.

The first row determines each target's shape. If that row fails, the program
cannot create the arrays and stops the job.

For later failures:

1. Read the printed exception and correct the scientific or numerical cause.
2. Keep the complete result set together.
3. Resume with `--loadchk 1 --append 0` and the same names and YAML.
4. Rerun the failure-file and shape check.
5. Give the files to a trainer only after every flag is `0`.

## FAQ C2. How do I resume a stopped run? <a id="faq-c2-resume"></a>

Reuse the same YAML, output names, `--unif`, `--temp`, `--maxcorr`, and
`--boundary` choices. Supply the fresh-run seed recorded in the `.1.txt`
header as well so the recovery command preserves a consistent record. Then
change only the run controls:

```text
--loadchk 1 --append 0
```

Resume loads the existing parameter rows; it does not redraw them from the
seed. The loader requires every expected parameter, failure, and physical
file, plus coordinate files for the families that write them. It checks row
counts and declared parameter columns.

The loader also requires the covariance, ranges, and facts files, but does not
repeat every scientific validation of their contents. If a required file is
missing or invalid, the command stops without treating the request as a fresh
run. A successful resume recalculates only rows whose failure flag is `1`.

Keep a copy of the full result set before recovery work. The files are saved
one after another, so an interruption can leave files from different save
moments.

## FAQ C3. How do I add more rows? <a id="faq-c3-append"></a>

Load the finished result set and request the extra rows:

```bash
--loadchk 1 --append 1 --nparams 5000
```

with the same names, YAML, and seed as the original run. The generator loads
the saved rows, draws `--nparams` new parameter rows from a stream derived
from the seed plus the existing row count (so the new rows never repeat the
old ones), extends every saved file with the new rows, and then computes the
new data vectors. Rerunning the same append command reproduces the same
appended rows.

## FAQ C4. Which files in this folder are commands? <a id="faq-c4-program-files"></a>

Run these files by path:

| File | User action |
| --- | --- |
| `dataset_generator_lensing.py` | generate CosmoLike vectors |
| `dataset_generator_cmb.py` | generate CMB spectrum rows |
| `dataset_generator_background.py` | generate $H(z)$ and radial comoving-distance $\chi(z)$ rows |
| `dataset_generator_mps.py` | generate matter-power rows |
| `compute_cmb_covariance.py` | calculate one CMB experiment covariance |

Do not run these helper modules as commands:

| File | Purpose inside the programs |
| --- | --- |
| `generator_core.py` | shared options, sampling, file writing, and MPI work |

Generator runs write their files directly under `chains/`; there is no
hidden state beside them.
