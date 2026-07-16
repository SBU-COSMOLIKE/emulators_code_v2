# Small tests for data handling, training rules, and AI tools

The Python files in this folder check one behavior at a time. For example, one
test confirms that a parameter table with one row keeps the two-dimensional
shape expected by the training programs. Another test damages a saved progress
file inside a temporary folder and confirms that the loader refuses it without
changing the original files.

The AI development roles run a relevant test after making a related change and
again before handing the result to the Architect. A user can run the same
commands to see the result. A test answers one narrow question such as the two
examples above. A gate is a larger named check whose required result is stated
before it runs. A gate may run tests, compare a scientific result, or use
configured data or hardware. For example, the dataset-publication gate runs
the tests that prove an accepted generated dataset cannot be changed. The next
section explains why both levels are useful.

Most checks run on the central processing unit (CPU) without training a model.
Checks that need files create temporary copies and delete them afterward, so
they do not change training data or requests saved by the AI tools.

Some final decisions need a configured scientific installation, a graphics
processing unit (GPU), or several checks taken together. The
[`ai/gates/` guide](../gates/README.md) explains how those decisions are made.

## Contents

1. [Why are tests and gates different?](#why-are-tests-and-gates-different)
2. [Run the usual test set](#run-the-usual-test-set)
3. [Choose a test, reproduction, or gate](#choose-a-test-reproduction-or-gate)
4. [Read the result](#read-the-result)
5. [What do these files use or change?](#what-do-these-files-use-or-change)
6. [Find the file that covers a behavior](#find-the-file-that-covers-a-behavior)
7. [Scientific and data test inventory](#scientific-and-data-test-inventory)
8. [AI workflow and policy test inventory](#ai-workflow-and-policy-test-inventory)
9. [Direct reproduction inventory](#direct-reproduction-inventory)
10. [How agents use these files](#how-agents-use-these-files)
11. [Add or update a test](#add-or-update-a-test)

## Why are tests and gates different?

A **test** checks one small behavior. For example, one test can confirm that a
table with one row keeps its two-dimensional shape. After changing the code
that loads a one-row table, a developer can run that test before starting the
next change. If it fails, the recent edit and the narrow question give a small
set of places to inspect.

A **gate** checks several related results together. For example, the
dataset-publication gate checks that accepted generated-data files remain
read-only, damaged files are refused, and later work starts from separate
writable copies. The gate runs the tests for those results and checks that no
required test disappeared. If one required result needs a GPU or configured
scientific data, the gate records that missing requirement as `UNAVAILABLE`
rather than calling an unperformed check a pass.

The distinction is similar to work in a laboratory. One measurement checks
one part of an experiment. Before accepting the experiment's result, the
researcher confirms that every required measurement was made under the stated
conditions. A test is one measurement; a gate checks the required set. A gate
often calls tests from this folder, so the two systems support each other.

Keeping both levels has two practical benefits:

- A failed test identifies a small problem quickly.
- A gate prevents a final decision from using only the convenient tests while
  omitting required data, hardware, or failure cases.

## Run the usual test set

Activate the Python environment used by the emulator library. It must provide
the repository's NumPy 1.x, PyTorch, h5py, PyYAML, and psutil dependencies. The
top-level [library guide](../../README.md) points to the environment setup.

Then open a terminal in the repository root, the folder that contains `ai/`
and `emulator/`:

```bash
python3 -m unittest discover -s ai/tests -p 'test_*.py'
```

Python finds every immediate file whose name starts with `test_`. A successful
run exits with code 0 and ends with `OK`. The number after `Ran` changes when a
test is added, so do not use a fixed count as the success condition.

To run one module:

```bash
python3 -m unittest ai.tests.test_parameter_table
```

To run one test method:

```bash
python3 -m unittest \
  ai.tests.test_parameter_table.ParameterTableTest.test_one_row_stays_two_dimensional
```

These commands import the current working source. A Git **commit** is one
saved project version, and the Git branch named `main` holds accepted
versions. The commands are useful before a commit because they check the
current edits, including work Git has not saved yet. Tests that need files
create temporary data as explained below.

## Choose a test, reproduction, or gate

A **regression test** reruns a small case that must keep working. A **test
module** is one Python file imported by the test runner.

A **reproduction** is a stand-alone Python script that rebuilds a tool failure,
often with temporary Git folders, files, locks, or simultaneous operations. A
file lock lets only one running program own shared work at a time. A **gate**
is a named validation job whose required result is written before it starts.
The validation board lists and runs those gates.

| Need | Location | Normal command | What it may require |
| --- | --- | --- | --- |
| Check one Python, numerical, file, or policy rule | A `test_*.py` file in this folder | `python3 -m unittest ai.tests.test_parameter_table` | CPU and the project Python environment |
| Rebuild a mailbox, Git, archive, or process failure | A `*_repro.py` file in this folder | `python3 ai/tests/tools_mailbox_daemon_role_models_repro.py` | CPU, temporary disk space, and sometimes the Git program |
| Accept a result that needs configured scientific data or named hardware | [`ai/gates/`](../gates/README.md) | `python3 ai/gates/run_board.py --gate finetune-smoke` | A configured CosmoLike installation and a supported GPU, as the gate guide explains |

The `unittest discover` command does not run the reproduction scripts because
their names do not start with `test_`. Run a reproduction only when the
source note or the part of the code being changed names it. The **source note** is the
Markdown file that records the requested change and required checks. Passing
this folder's tests does not replace a gate required by that note.

## Read the result

An automatically found test prints dots while cases pass. A successful run
ends with a line beginning with `Ran` and then a final line containing `OK`.

An **exit code** is the number a command returns to the terminal: 0 means the
command succeeded, and any other value means it did not. `FAILED`, `ERROR`, or
a nonzero exit code means the change is not accepted. Read the first
**traceback**, the block that names the file, test method, and failing
comparison. Later failures may be consequences of that first one.

Reproduction scripts print several `PASS` or `FAIL` lines. Their exact summary
wording differs, but all current scripts return exit code 0 only when every
required check passes. For example:

```bash
python3 ai/tests/tools_mailbox_daemon_role_models_repro.py
```

The final `runtime-summary` must report no failures, and the command must exit
with code 0.

A passing test supplies evidence. It does not issue the Architect's final
`GO`. The [AI development guide](../README.md) explains that decision step.

## What do these files use or change?

All current files in `ai/tests/` run on the CPU. CUDA is NVIDIA software for
running calculations on a GPU. A mention of CUDA in a test name, test input,
or expected message does not start a CUDA job. Live GPU training belongs to
the gate board.

The scientific tests import the same NumPy, PyTorch, h5py, and YAML code used
by the library. They use small arrays or model fragments. They do not need a
Claude or Codex account, and they do not launch an AI role.

Many tests create a `TemporaryDirectory`. Mailbox reproductions place their
requests, locks, Git folders, and saved work-folder records inside that
temporary directory. They may read the library files in the current working
folder, but their deliberately damaged source copies remain in memory or in
the temporary folder. They do not write to `ai/notes/mailbox/` in the working
repository.

The operating system removes a `TemporaryDirectory` after a normal exit. If a
process is forcibly killed, a temporary folder may remain under the system
temporary directory. It is not a project result and may be removed after no
test process is using it.

The longest reproduction files exercise many failure cases, several child
programs running at once, and deliberately damaged source copies. Tests of
saved work folders, fix-only runs, runs without a Red Team, stopping cycles,
source changes during a run, and manual message routing can take longer than
one focused `unittest` module. Run them individually before combining their
output with another command.

## Find the file that covers a behavior

Search the full folder from the repository root:

```bash
rg -n "parameter table" ai/tests --glob '*.py'
rg -n "skip-redteam" ai/tests --glob '*.py'
```

`rg` prints each matching filename and line number. Exit code 1 means that no
line matched; it does not mean Python ran a test.

Use the tables below when the search term appears in several files. The tables
under “Scientific and data test inventory” and “AI workflow and policy test
inventory” list modules found by `unittest discover`. The table under “Direct
reproduction inventory” lists scripts that must be called directly. The
detailed scientific entries name the example input, the action performed, the
required result, a refused case, and why that result matters.

## Scientific and data test inventory

This section covers tests for scientific input files, generated training data,
training calculations, and saved emulators. The short tables only help find
the file that answers a question. The explanations below each table then show
a real input, the action performed, the required result, one case that must
stop, and why that result matters.

A **checkpoint** is the set of files saved so an interrupted data-generation
run can continue. A **SHA-256 fingerprint** is calculated from the exact bytes
in a file; changing one byte changes the fingerprint. Grid2D data form a
redshift-by-wavenumber surface.

### Input tables and generated training data

| File | Question answered |
| --- | --- |
| `test_background_grid_contract.py` | Do training setup and the Cobaya adapter accept the same background quantities and physical units? |
| `test_cmb_checkpoint_axis.py` | Can each saved CMB spectrum column be matched to the correct multipole `ell` value before any spectrum is loaded? |
| `test_data_staging_paramnames.py` | Does each numeric parameter column receive the correct physical name before its data-vector row is opened? |
| `test_dataset_publication.py` | Can readers see only one complete, unchanged generated dataset while resume or append work uses separate writable copies? |
| `test_dataset_request_contract.py` | Does a saved request describe the exact scientific calculation and the exact files that calculation must produce? |
| `test_generator_checkpoint_refusal.py` | Does an explicit resume or append stop when any required progress file is missing or damaged? |
| `test_generator_member_binding.py` | Does each generator determine its complete, safe list of filenames before touching the filesystem? |
| `test_generator_payload_success.py` | Is a generated row marked successful only after its scientific values survive validation, writing, and exact read-back? |
| `test_generator_run_control.py` | Do the three legal run choices select new work, resume, or append without allowing one output kind to overwrite another? |
| `test_grid2d_staging_row_contract.py` | Do in-memory and disk-backed Grid2D inputs select the same scientific rows in the same order? |
| `test_parameter_table.py` | Are sampled and derived parameter columns selected by name instead of by a remembered column number? |

#### Background quantities and physical units

`test_background_grid_contract.py` uses three artificial measurements of the
Hubble rate or transverse distance at two redshifts.

- **Example used:** the accepted pairs are `Hubble` with `km/s/Mpc`, and
  `D_M` with `Mpc`. The example also supplies the offset used before an
  optional logarithm.
- **What the test does:** it validates the training settings, constructs and
  rebuilds the numerical transformation, and initializes a small Cobaya
  adapter that reads the same list of accepted quantity-and-unit pairs.
- **Pass means:** the requested quantity and unit stay paired, the offset is an
  ordinary finite number, every training value remains finite through the
  transformation, and the saved offset, center, and scale are finite when the
  transformation is rebuilt.
- **A refusal it proves:** `Hubble` with `Mpc`, an unknown unit, `NaN`,
  infinity, a Boolean, text in place of the offset, or a `NaN` or infinite
  saved center or scale must stop before background values are used.
- **Why it matters:** arrays can have the expected shape while still carrying
  the wrong physical units. Such an array can produce a finite but
  scientifically incorrect prediction.

#### CMB multipole labels

`test_cmb_checkpoint_axis.py` checks the coordinate labels saved beside the
four CMB spectra `TT`, `TE`, `EE`, and `PP`.

- **Example used:** four spectrum columns represent
  `ell = 2, 3, 4, 5`. The companion NumPy file must contain those exact
  consecutive integers using the `int64` format.
- **What the test does:** it first asks the loader to verify the coordinate
  file, and records whether the loader tries to open any spectrum.
- **Pass means:** the exact coordinate file is accepted, all four spectra load,
  and none of the checkpoint files change.
- **A refusal it proves:** a missing coordinate file, `int32` storage, a
  two-dimensional array, the wrong length, a shifted range, reversed values, a
  gap, or a changed order must stop before the first spectrum is read.
- **Why it matters:** without this check, a valid number in one spectrum
  column could be interpreted as the power at the wrong multipole.

#### Parameter names used while selecting training rows

`test_data_staging_paramnames.py` checks how a numeric table is connected to
the file that names its columns. Selecting and copying the rows needed for
training is called **staging** in this library.

- **Example used:** `chain.1.txt` may use `chain.1.paramnames` when it
  exists, and otherwise may use `chain.paramnames`. A plain file such as
  `lcdm.v2.txt` uses `lcdm.v2.paramnames`; `v2` is part of its name, not
  a chain number.
- **What the test does:** it resolves names for sampled parameters and a
  derived `chi2*` column before a scalar, CMB, Grid, or Grid2D loader opens
  the matching data vector.
- **Pass means:** the exact or allowed shared names file is chosen, all
  requested columns are found by name, and every loader agrees on the number
  of usable rows.
- **A refusal it proves:** absent or reordered sampled names stop the operation
  before the data-vector file is opened.
- **Why it matters:** opening the data vector first could pair one cosmological
  parameter name with another column's numbers.

#### Publishing one complete generated dataset

`test_dataset_publication.py` checks how a finished group of generated files
becomes the group that readers use. A **generation** is one complete named
group. A small active-record file names the currently selected generation.

- **Example used:** the test publishes nested files as generation A, publishes
  generation B, and opens readers before and after the change. It also starts
  a writable continuation from A.
- **What the test does:** it records every required path, byte count, and
  SHA-256 fingerprint; makes the accepted files read-only; and then simulates
  changed bytes, extra files, links, interrupted copies, failed disk
  synchronization, cleanup failures, and two writers finishing in a different
  order.
- **Pass means:** a reader that already opened A keeps reading A, a later
  reader sees complete B, and continuation files are separate writable copies
  whose edits cannot change A. An older writer cannot replace a newer active
  record.
- **A refusal it proves:** missing or extra files, changed bytes, symbolic or
  hard links, a filename that leaves the generation folder, a file changed
  during copying, or a stale writer must stop without changing the usable
  active generation.
- **Why it matters:** training must never combine some files from A with other
  files from B, nor use files that changed after they were accepted.

#### The saved description of a requested dataset

`test_dataset_request_contract.py` checks the record that says exactly which
scientific dataset a generator was asked to produce.

- **Example used:** records cover uniform and Gaussian sampling, the random
  seed, ordered parameter names, full or chain-only output, configuration
  fingerprint, generator family, and matter-power variant. The **probe** is
  the observable being requested, such as cosmic shear, CMB, background
  quantities, or matter power.
- **What the test does:** it converts each record to a fixed JSON byte
  sequence and separately lists the required output files, such as four CMB
  spectra plus their multipole coordinates or matter-power grids plus their
  coordinate arrays.
- **Pass means:** the same scientific request always gives the same JSON bytes
  and file list. Changing a scientific choice, including the seed, parameter
  order, sampling temperature, probe, or variant, changes the record.
- **A refusal it proves:** duplicate parameter names, a uniform-sampling
  boundary factor or Gaussian maximum-correlation value outside its allowed
  range, an incompatible family, probe, generator, or variant, or a field
  belonging to the other sampling method must stop. Checkpoint frequency and
  other write-management choices must not be inserted as if they changed the
  science.
- **Why it matters:** a resume must continue the same calculation, not a
  different calculation that happens to use similar filenames and shapes.

#### Missing or damaged progress files

`test_generator_checkpoint_refusal.py` checks the files used to continue an
interrupted data-generation run.

- **Example used:** it creates complete, incomplete, and damaged progress
  groups for lensing, CMB, background, and matter-power generation.
- **What the test does:** it requests a new run, a resume, or an append and
  watches whether sampling begins. It also checks full-data and chain-only
  requests separately.
- **Pass means:** a new run may correctly report that no progress exists. A
  requested resume or append loads only after every required parameter file,
  failure marker, data file, and coordinate array for that generator passes
  its checks. Chain-only work requires only its parameter-chain files.
- **A refusal it proves:** a missing CMB multipole file, a coordinate array
  with the wrong length, an invalid failure marker, or any other missing
  required member stops instead of silently starting again from row zero.
- **Why it matters:** silently starting over could overwrite useful progress
  or combine old and newly generated rows under one filename.

#### The filenames owned by each generator

`test_generator_member_binding.py` checks the complete file list assigned to
each scientific generator.

- **Example used:** lensing, CMB, background, and matter-power generators are
  tested in full-data and chain-only modes. The optional Syren matter-power
  base adds exactly two base arrays.
- **What the test does:** after validating the settings, it asks the generator
  for every absolute path it may use and compares that list with the files the
  generator's real save and load methods expect. No candidate file is opened
  during this step.
- **Pass means:** each generator receives only its own family and variant
  files, full and chain-only names remain separate, and the saved file list is
  reused by later operations.
- **A refusal it proves:** a family sent to the wrong generator, an
  incompatible family variant, output names that resolve to different
  folders, two names differing only by letter case, a driver whose file list
  cannot be verified, or a non-Boolean Syren switch stops before the program
  checks whether any path exists.
- **Why it matters:** similar filenames must not let one generator read files
  that belong to another scientific calculation.

#### When a generated row becomes successful

`test_generator_payload_success.py` checks the point at which one generated
sample changes from failed to successful.

- **Example used:** small flat vectors, four CMB spectra, two background
  quantities, and matter-power output with and without the optional base are
  saved in temporary progress files.
- **What the test does:** it validates a row, converts it to the configured
  storage format, writes it once, reads it back, and compares the stored bytes
  with the validated values. Existing rows already marked successful or
  failed are also checked during resume.
- **Pass means:** the failure marker clears only after names, shape, finite
  numbers, storage type, and exact read-back values all agree. A saved row
  already marked failed stays failed.
- **A refusal it proves:** `NaN`, infinity, conversion overflow, a missing
  background quantity, a wrong CMB spectrum length, an unexpected
  matter-power quantity, or changed bytes after writing leaves the row failed
  and does not silently repair an existing file.
- **Why it matters:** an interrupted or overflowing calculation must not be
  counted as valid training data merely because some bytes were written.

#### New work, resume, append, and chain-only output

`test_generator_run_control.py` checks the three command settings
`loadchk`, `append`, and `chain`.

- **Example used:** the test tries every combination of omitted values, the
  exact integers `0` and `1`, and look-alikes such as `True`, `1.0`,
  and the text `"1"`.
- **What the test does:** it turns the first two settings into one operation
  and the third setting into either full-data or chain-only output. It then
  asks for the filename beginning used by that output.
- **Pass means:** `(0, 0)` selects new work, `(1, 0)` selects resume, and
  `(1, 1)` selects append. Chain-only output adds `_chain_only`, and the
  validated decision cannot later be edited.
- **A refusal it proves:** append without resume, a value other than an exact
  integer `0` or `1`, or an empty or unknown output choice stops before a
  generator file changes.
- **Why it matters:** these checks prevent an accidental fresh run or a
  chain-only request from overwriting a complete generated dataset.

#### Selecting Grid2D rows from memory or disk

`test_grid2d_staging_row_contract.py` checks that two storage methods select
the same scientific samples.

- **Example used:** seven matter-power rows are stored once fully in memory
  and once in a NumPy file opened a few rows at a time. With the fixed seed,
  both routes must select source rows `5, 1, 6, 2` in that order.
- **What the test does:** it selects and converts both the raw surface and its
  matching base surface, then compares parameters, target values, and order
  from the two routes.
- **Pass means:** both routes return the same four rows in the seeded order.
  The raw and base files each contain exactly the declared seven rows before
  memory for the converted result is allocated.
- **A refusal it proves:** a file with six or eight rows, or a row count given
  as `True`, zero, a negative number, text, or a NumPy integer stops before
  allocating the converted array.
- **Why it matters:** a changed order or incorrect row count could pair one
  cosmology's parameters with another cosmology's power spectrum.

#### Reading sampled and derived parameter columns

`test_parameter_table.py` checks GetDist-style numeric tables and their
companion `.paramnames` files. A trailing `*` in that names file marks a
quantity calculated from the sampled parameters.

- **Example used:** one-row and multirow tables contain two GetDist
  bookkeeping columns followed by sampled and derived columns. Numbered chains,
  reordered requested outputs, and a UTF-8 names file are included.
- **What the test does:** it chooses the allowed names file and extracts input
  and output columns by name rather than by a remembered position.
- **Pass means:** one row remains a two-dimensional table, zero requested
  outputs have shape `(rows, 0)`, derived outputs may be requested in a new
  order, and returned values use `float32`.
- **A refusal it proves:** duplicate or missing names, an invalid `*` marker,
  overlap between inputs and outputs, a table one column too short or long, an
  empty table, `NaN`, or infinity stops with an error naming the unsafe
  mapping. A missing names file lists the paths tried instead of guessing.
- **Why it matters:** selecting by a remembered column number can silently
  assign a parameter's values to the wrong physical name.

### Training calculations

| File | Question answered |
| --- | --- |
| `test_batching_sizing.py` | Does the batch planner count the actual memory used by every stored target value before training starts? |
| `test_d5_training_behavior_witnesses.py` | Do small numerical examples support the learning-rate, moving-average, activation, and frozen-layer results required by the longer training gate? |
| `test_finetune_post_step_and_provenance.py` | Does fine-tuning update one weight in the required order and save which earlier emulator supplied the starting weights? |
| `test_trf_token_width.py` | Does a Transformer refuse a one-number token that cannot respond to its input while continuing to accept supported two-number tokens? |
| `test_warmstart_perturbed_finite.py` | Does a warm start report the exact input or output that first becomes `NaN` or infinite? |

#### Memory needed for one training batch

`test_batching_sizing.py` checks how much memory is needed for the group of
rows processed together during one training step.

- **Example used:** three samples each have two input parameters. An ordinary
  target stores seven numbers per sample; a packed target stores fourteen.
  Both `float32` and `float64` storage are tried.
- **What the test does:** it counts the bytes used by inputs, model outputs,
  targets, and temporary calculations, then asks how many complete batches fit
  in a stated memory allowance.
- **Pass means:** `float64` target values use twice the target bytes of
  `float32`, the packed target changes the chosen batch count, and the
  streaming loader uses the same width and number format in its calculation.
- **A refusal it proves:** if the allowance is even four bytes short of one
  complete batch, the planner raises `MemoryError` and reports the byte
  breakdown instead of forcing one batch that does not fit.
- **Why it matters:** starting with a batch that cannot fit can terminate a
  long training job after its data have already been prepared.

#### Small numerical examples for training behavior

`test_d5_training_behavior_witnesses.py` supplies CPU examples for several
results that the longer training gate must later observe. BerHu is an optional
rule for calculating training loss. A moving average is a smoothed record of
recent model weights or measurements.

- **Example used:** it creates a smooth BerHu-loss schedule, a moving-average
  schedule, tiny ReLU and Tanh networks, a complete 30-epoch learning-rate
  record, and fingerprints calculated from the earlier shared network layers.
- **What the test does:** it checks selected epochs and both points where each
  schedule changes behavior, trains the small networks on a nonlinear table,
  recomputes the first moving-average coefficient, reads the learning-rate
  record, and compares shared-layer fingerprints before and after two kinds of
  training.
- **Pass means:** the schedules rise at the stated epochs without a jump; ReLU
  and Tanh learn more than a model that always predicts the mean; the
  final-layer learning rate changes exactly once at epoch 20; the first live
  moving-average record is complete; and shared layers change only when all
  layers are allowed to learn.
- **A refusal it proves:** a constant or early schedule, a network whose loss
  does not decrease, `NaN` loss, a missing or duplicate epoch, an incorrect
  moving-average coefficient, or a changed supposedly frozen shared layer
  makes the test fail.
- **Why it matters:** printed labels such as “changed” or “frozen” are not
  accepted on trust; the test recalculates the numerical result.

#### Order of operations after a fine-tuning step

`test_finetune_post_step_and_provenance.py` uses a model with one adjustable
number so the required order can be calculated by hand.

- **Example used:** a one-weight arithmetic example starts at 4. An anchor
  operation uses the optimizer's learning rate to move the weight to 3, and a
  moving average configured to copy the current value must also become 3.
- **What the test does:** it reads the production training loop and confirms
  that its optimizer, anchor, and moving-average calls occur in that order.
  The separate arithmetic example executes the anchor and moving average. The
  test also checks that the cosmic-shear and scalar training drivers call one
  shared function to save the starting-emulator description.
- **Pass means:** the moving average includes the already anchored weight, and
  both drivers save the same concrete facts about the earlier emulator that
  supplied the starting weights.
- **A refusal it proves:** moving the anchor call before the optimizer or after
  the moving-average call fails the order check. The test also fails if either
  training driver stops using the shared function that writes the source
  description.
- **Why it matters:** operation order changes the trained weight, while the
  saved source facts let a later reader identify which emulator the run
  started from.

#### Transformer token width

`test_trf_token_width.py` checks how many numbers a Transformer places in
each token, a small group of related numbers processed together. Plain and
factored are the two supported Transformer layouts in this library.

- **Example used:** tiny plain and factored Transformers produce either one
  number or two numbers per token.
- **What the test does:** it tries both layouts with one and two numbers per
  token, and runs the older one-number calculation on two different inputs.
- **Pass means:** one-number tokens stop before a model is built. Supported
  two-number tokens build normally and produce different corrections for
  different inputs.
- **A refusal it proves:** the older one-number calculation gives the same
  correction to two different inputs. Removing the early refusal therefore
  makes this test fail.
- **Why it matters:** a block whose output does not depend on its input cannot
  learn the requested relationship, even though its arrays have legal shapes.

#### Invalid numbers during a warm start

`test_warmstart_perturbed_finite.py` checks fine-tuning and transfer learning
that begin from an earlier saved emulator rather than random weights.

- **Example used:** twelve artificial parameter rows are prepared, with source
  rows 4 and 9 selected. One added parameter changes from zero to one. A
  fixture can create a `NaN` while row 9 is converted to network input, or
  infinity in row 9's initial fine-tune model output or transfer-composed
  prediction.
- **What the test does:** both warm-start routes compare the unchanged and
  perturbed calculation after parameters are converted into the numbers read
  by the network and after the initial model or combined prediction is
  produced. The test temporarily bypasses each finite-number check in memory
  to prove which check detects the problem.
- **Pass means:** finite control rows preserve the earlier numerical result.
  Invalid input or output reports source row 9 and the exact named quantity
  that first became nonfinite.
- **A refusal it proves:** without the early input or output check, the older
  code later blames a different comparison. The temporary removal must
  reproduce that misleading diagnosis and make the test fail.
- **Why it matters:** the first invalid number identifies the real
  transformation that needs attention; a later error can point to the wrong
  scientific quantity.

### Saved emulator files and reconstruction

An emulator result is stored in an HDF5 file. HDF5 divides one file into named
sections, similar to named folders inside the file. Rebuilding means opening
that saved result and constructing the model needed for new predictions.

| File | Question answered |
| --- | --- |
| `test_grid2d_const_mask.py` | Does a saved Grid2D geometry remember exactly which coordinates use fixed stored values instead of neural-network predictions? |
| `test_results_composition_mode.py` | Does the result file state how its neural-network output and any saved base are combined into a physical prediction? |
| `test_results_const_mask_declaration.py` | Can a reader detect one changed Grid2D fixed-coordinate position after the result was saved? |
| `test_results_rebuild_fixed_facts_names.py` | Does reopening an emulator stop when saved input names disagree, even if the structured scientific record and its saved text copy were changed together? |

#### Fixed coordinates in a Grid2D surface

`test_grid2d_const_mask.py` checks a one-dimensional mask stored in the
coordinate order of a two-redshift by three-wavenumber surface. A `1` means
“use the saved fixed value here”; a `0` means “use the network prediction.”

- **Example used:** one mask has no fixed coordinates. A second mask fixes the
  lowest-wavenumber point at each redshift.
- **What the test does:** it saves the geometry, rebuilds it, applies a toy
  network output, and compares the result with the original surface rule.
- **Pass means:** even the all-zero mask is saved explicitly. The two fixed
  positions in the second mask survive saving and loading and use the exact
  stored value.
- **A refusal it proves:** an absent mask, a two-dimensional mask, the wrong
  stored number format, or a value other than zero or one stops instead of
  being guessed.
- **Why it matters:** guessing that a missing mask means “no fixed
  coordinates” can change the reconstructed matter-power surface.

#### How a saved model forms its final prediction

`test_results_composition_mode.py` checks the explicit rule used to combine a
network output with any saved base. Plain mode uses the network directly.
Neural polynomial chaos expansion (NPCE) adds a saved polynomial-chaos base.
Transfer mode uses an earlier emulator, either unchanged or refined.

- **Example used:** four in-memory result files represent plain output, NPCE,
  unchanged transfer, and refined transfer. Each contains the saved YAML
  settings and the named HDF5 sections required by that form.
- **What the test does:** the writer, reader, rebuild path, and prediction path
  must agree on the explicit mode, whether transfer was refined, and the
  required sections. The test watches whether geometry or weight loading
  begins before validation finishes.
- **Pass means:** exactly the four supported combinations pass, and the
  prediction path selects its additional data from the saved mode rather than
  guessing from whichever HDF5 section is present.
- **A refusal it proves:** a missing or unknown mode, text `"False"` in
  place of a Boolean, a missing required section, a section belonging to
  another mode, or disagreement between YAML and the saved attributes stops
  before geometry construction or weight loading.
- **Why it matters:** different forms can have weights with identical shapes
  but require different mathematics to produce a physical prediction.

#### Fingerprint for the Grid2D fixed-coordinate mask

`test_results_const_mask_declaration.py` checks a SHA-256 fingerprint
calculated by the result writer from the complete mask in coordinate order.

- **Example used:** two masks have the same number of fixed positions, but the
  `1` occurs at a different coordinate in each mask.
- **What the test does:** it calls the writer's fingerprint calculation,
  stores the result in a small in-memory HDF5 section, and then calls the
  reader's validator to recalculate the fingerprint from the stored mask.
- **Pass means:** the saved fingerprint matches the exact ordered mask. Moving
  one fixed position produces a different fingerprint.
- **A refusal it proves:** a caller cannot replace the writer's fingerprint;
  changing a mask position after saving, attaching the declaration to a
  non-Grid2D result, or opening an older Grid2D result with no declaration
  stops and requests a new save.
- **Why it matters:** merely counting fixed positions would miss a mask moved
  to a different physical coordinate.

#### Input names checked before model weights load

`test_results_rebuild_fixed_facts_names.py` checks independent records of the
input parameter order inside one saved emulator.

- **Example used:** the saved geometry says `alpha, beta`. A damaged example
  changes the structured scientific record and its saved text copy together
  to `beta, alpha` but leaves the geometry unchanged.
- **What the test does:** it reopens the emulator and records whether PyTorch
  is asked to load model weights.
- **Pass means:** an untouched file whose names agree reaches weight loading.
  The test stops there because reaching that point proves that a valid file
  was not rejected.
- **A refusal it proves:** those two coordinated changes still disagree with
  the geometry and must stop before model weights are requested.
- **Why it matters:** otherwise the model can receive valid finite numbers in
  the wrong parameter order and return a scientifically incorrect prediction.

## AI workflow and policy test inventory

These modules check the rules used by the Architect, Implementer, Red Team,
mailbox watcher, and repository protection tools.

| File | What it checks |
| --- | --- |
| `test_backlog_guard.py` | Creates a small local backlog and records the SHA-256 fingerprint of its exact bytes, just as the Architect does before and after an edit. An unchanged backlog must pass; a changed backlog must fail until the Architect supplies the previously accepted fingerprint and saves the new one. The test also proves that an Implementer or Red Team cannot authorize that update, and that a missing file, link, hard link, directory, oversized file, malformed saved record, or file changed during reading is refused instead of being mistaken for an approved backlog. |
| `test_handoff_contract.py` | Writes sample Architect and Red Team instruction files, then passes them to the same validator used by the mailbox tools. A valid file must contain the ordered plan, exact files and commands, evidence destination, severity decision, and character limit; missing sections, placeholders, hidden Markdown, an oversized or non-UTF-8 note that holds the ticket details, or a different validator program must refuse without changing the instruction file. |
| `test_mailbox_conditional_preamble.py` | Builds one prompt for a role that must send work onward and one terminal prompt for a role whose job ends there. Only the first may require a reply message; the terminal prompt must say that no reply is required and must not contain a second general instruction that contradicts that ending. |
| `test_mailbox_daemon_architect_entrypoint.py` | Runs public `send` and `ping` commands in a temporary mailbox and inspects the files they create. A user request must create exactly one message to the Architect with the original text and chosen severity; public commands that address the Implementer or Red Team must refuse with no new file, while the watcher may still deliver the Architect's internal messages to those roles. |
| `test_mailbox_daemon_severity.py` | Sends discovery requests with no severity and with `high`, `medium`, or `low`, then records which value each started role receives. Missing values become `medium`, malformed headers never launch work, and fix-only mode or a disabled Red Team must remain stronger than a request to search at low severity. |
| `test_permanent_note_guard.py` | Copies the eleven permanent notes and their guard into a temporary Git repository, records the approved project version, and then changes one boundary at a time. The guard must pass the untouched copy but refuse edited current files, edits chosen for the next commit, edits already committed, an extra tracked note, a note replaced by a link, a changed guard program, or a shortened starting-version identifier. |
| `test_permanent_note_style_contract.py` | Reads every permanent note as text and checks rules that make the notes useful after the current ticket is gone. The files must use neutral present-tense language, avoid dates and personal diary labels, provide unique stable link targets, and never depend on a temporary audit or backlog file. |
| `test_role_directive_contract.py` | Reads the role templates, mailbox prompts, validator text, and reader guide together. It checks that the user speaks only to the Architect, the Architect and Red Team provide detailed repair steps and evidence, the Implementer stops rather than inventing a design, character limits never excuse unreadable code, and neither the Implementer nor Red Team may edit permanent notes. |
| `test_tests_readme_inventory.py` | Lists every immediate `.py` file in `ai/tests/` and extracts the backticked filenames from these inventory tables. The two sets must be identical, so adding a test without explaining it here, documenting a file that does not exist, or listing the same file twice makes the test fail. |
| `test_ticket_change_guard.py` | Creates small Git histories containing added, deleted, renamed, replaced, Unicode, binary, and non-UTF-8 files, then applies the ticket's `--max` character limit. A positive limit must count added plus deleted characters from the exact clean starting version and refuse hidden, oversized, binary, or unreadable changes; zero must mean no size limit, and measuring must not change the list of files already chosen for the next commit. |

## Direct reproduction inventory

These scripts are not found by `unittest discover`. Each script has a
`main()` starting function, prints its own checks, and returns a nonzero exit
code when its witness fails.

| File | What it rebuilds inside temporary folders |
| --- | --- |
| `finite_contract_cuda_wording_repro.py` | Runs the finite-number gate on a machine that cannot perform its required CUDA compilation. The gate must report that this machine check is still unavailable and name the missing CUDA action; it must not print wording that could be mistaken for a scientific pass. |
| `tools_backlog_bundle_repro.py` | Creates a backlog and supporting files, packs them into `.tar.xz`, imports that archive into a new temporary folder, and compares every byte. It also tries an existing destination, changed protected files, links, names that escape the destination, malformed archives, and oversized inputs; each unsafe case must stop without replacing the receiver's files. |
| `tools_handoff_router_repro.py` | Creates a temporary Git repository and sends sample Architect, Implementer, and Red Team handoffs through the manual router. It checks the selected working folder, message order, lock cleanup, clipboard option, detailed directive and evidence sections, severity and character-limit values, and refusal of a wrong role, bad source note, or malformed instruction file. |
| `tools_mailbox_daemon_dead_mailbox_repro.py` | Starts with no watcher, then with live, held, and outdated lock files, and runs preview, send, and ping commands. The user must receive the correct warning without a blocking read; a second live watcher, a required folder replaced by a link, or a pipe file that could wait forever must refuse safely. |
| `tools_mailbox_daemon_fix_only_repro.py` | Places one repair request and one request to discover a new problem in a temporary mailbox, then starts the watcher with fix-only mode. Only the repair may start, simultaneous attempts to enable the mode must agree, each role must receive the rule, two-role operation must still work, and changed copies of the daemon or disagreement between `--help` and the README must be detected. |
| `tools_mailbox_daemon_landing_debt_repro.py` | Builds a history where a fix was accepted in a role's branch but never reached `main`, then runs repeated watcher passes. Exactly one repair request must be created until the fix lands; held locks and malformed saved records must not duplicate the request, and a corrected record must let later passes recover. |
| `tools_mailbox_daemon_max_repro.py` | Starts tickets with an omitted, zero, positive, malformed, and conflicting `--max` value. The default must be zero, a positive value must appear in the terminal and in every role's environment, and malformed or disagreed values must stop before a role starts. |
| `tools_mailbox_daemon_no_redteam_repro.py` | Starts a watcher with the Red Team disabled while Architect and Implementer requests are waiting. Those two roles must use separate saved work folders, Red Team requests must remain waiting, fix-only mode and simultaneous sends must stay safe, locks must be removed at the end, and a changed daemon copy must refuse. |
| `tools_mailbox_daemon_output_style_repro.py` | Runs ordinary waits, refusals, and progress reports and compares the exact terminal text with the examples in the README. The check catches unclear separator lines, unexplained all-capital messages, wrong waiting counts, or documentation that shows output the program no longer prints. |
| `tools_mailbox_daemon_primary_worktree_repro.py` | Uses temporary Git repositories to create and then reuse one separate Claude work folder and one separate Sol work folder. It checks the saved folder records, simultaneous first-time setup, movement of older mailbox files into the selected folders, and refusal when an existing branch or folder name would point at the wrong place. |
| `tools_mailbox_daemon_redteam_repro.py` | Sends several Red Team messages while preview mode and competing watcher processes are active. A message must be completely written before it becomes visible, only one watcher may claim it, message numbers must remain unique, and malformed bodies or unreplaced placeholder text must refuse without losing another message. |
| `tools_mailbox_daemon_rendezvous_repro.py` | Starts short-lived fake roles and observes the occasional 20-second Ctrl-C window after all work already in progress finishes. It proves that this manual stopping opportunity is separate from a ticket cycle, checks that a new request arriving during the window remains safe, exercises two simultaneous senders, and requires refusal when the backlog file cannot be understood. |
| `tools_mailbox_daemon_role_models_repro.py` | Selects model names and aliases for the Architect and Implementer, then inspects the exact commands the watcher would launch for each role. Defaults, valid aliases, invalid names, and role routing must agree, and changing a copied daemon source file must stop the run rather than launching an unverified command. |
| `tools_mailbox_daemon_staleness_repro.py` | Starts fake role commands that finish, time out, retry, or leave a request in the `inflight/` folder, which holds a message while one role is working on it. It checks finished-file order and recovery, and it requires refusal when a program file changes during the run, a required file becomes a link, a timeout record is malformed, or a copied daemon no longer matches its approved bytes. |
| `tools_mailbox_daemon_ticket_cycle_repro.py` | Gives the Architect and Implementer several messages for one temporary ticket and proves that conversation alone does not finish a cycle. A normal cycle finishes only after a different descendant commit is accepted and the Red Team returns a review tied to that exact ticket and commit. Emergency checks require two different tickets and commits, one from each Implementer. They also prove that only already admitted emergency work may finish after the threshold clears and that a finished ticket with no admitted partner is saved without falsely counting a cycle. Restart checks preserve a completed-cycle return across a crash, recover a rejected Architect receipt, and replay an old completed receipt without depending on a Git object forever. False ticket names, changed modes, wrong role routes, reused or unrelated commits, a missing backlog, and attempts to start extra work after the requested cycle count all refuse. |

## How agents use these files

The Architect names the smallest relevant test command in the implementation
instructions. The command must distinguish the repaired behavior from the
old defect; a test that passes both versions is not enough.

The Implementer runs that focused command while editing, then runs the full
`unittest discover` command. If the change affects a tool whose stand-alone
reproduction appears above, the Implementer runs that script separately and
returns its raw summary and exit code.

The Red Team reviews only the named change unless the user asked the Architect
for a wider search. A Red Team finding should name the test or reproduction
that demonstrates the defect. The Architect reruns required commands and
issues the final `GO` or `NO-GO`.

Tests and reproductions never update the eleven permanent notes. The
Architect handles any required permanent-note update as a separate protected
step.

## Add or update a test

Use a `test_*.py` module for a small repeatable behavior:

1. Choose a filename that names the behavior rather than a ticket number.
2. Use `unittest.TestCase` and a method whose name states the expected result.
3. Keep arrays and models small enough for CPU use.
4. Create files under `TemporaryDirectory`; do not borrow the user's mailbox,
   training files, or Git work folder.
5. Prove the test detects the defect by running it against a deliberately
   damaged copy or by otherwise showing that the old behavior fails.
6. Add one row for the new file to the appropriate inventory table.
7. Run the focused module and the full discovery command.

Use a direct reproduction when the defect needs several processes, file
locks, a temporary Git repository, malformed or unusual file types, or exact
command-line output. Its `main()` must return 0 only after every check passes
and a nonzero value after any check fails. Print the failed condition instead
of only raising a bare assertion.

Add a gate only when acceptance needs live training, configured data, a
saved-log requirement, or named hardware. Register it in `ai/gates/board.py`
and follow the separate [gate guide](../gates/README.md).

The inventory test below is part of normal discovery. It fails when a Python
file is added or removed without updating this guide:

```bash
python3 -m unittest ai.tests.test_tests_readme_inventory
```
