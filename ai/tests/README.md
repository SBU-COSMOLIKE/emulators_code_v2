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
8. [Three kinds of AI workflow checks](#three-kinds-of-ai-workflow-checks)
9. [AI workflow and policy test inventory](#ai-workflow-and-policy-test-inventory)
10. [Direct reproduction inventory](#direct-reproduction-inventory)
11. [How agents use these files](#how-agents-use-these-files)
12. [Add or update a test](#add-or-update-a-test)

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
| `test_cmb_covariance_publication.py` | Can a completed CMB covariance receive its public filename without replacing an earlier result or exposing an interrupted write? |
| `test_data_staging_paramnames.py` | Does each numeric parameter column receive the correct physical name before its data-vector row is opened? |
| `test_dataset_locator.py` | Can a familiar parameter-chain filename find the current complete dataset without changing that filename after every publication? |
| `test_dataset_publication.py` | Can readers see only one complete, unchanged generated dataset while resume or append work uses separate writable copies? |
| `test_dataset_request_contract.py` | Does a saved request describe the exact scientific calculation and the exact files that calculation must produce? |
| `test_cocoa_dataset_resolution.py` | Do the training and validation filenames in a user YAML resolve to two complete, internally consistent generated datasets? |
| `test_failed_row_staging.py` | Are rows that the generator marked as failed removed before training rows are selected? |
| `test_generator_checkpoint_refusal.py` | Does an explicit resume or append stop when any required progress file is missing or damaged? |
| `test_generator_dark_energy_facts.py` | Does the generator recognize that sampled `w0pwa` and `w` make the calculated `wa` vary, then save the physical `w, wa` law? |
| `test_generator_ingress.py` | Does a generator validate every requested parameter, covariance entry, fiducial value, family grid, and final sample count before it creates output? |
| `test_generator_member_binding.py` | Does each generator determine its complete, safe list of filenames before touching the filesystem? |
| `test_generator_mpi_message_binding.py` | Can an MPI worker result update only the parameter row that rank zero actually assigned to that worker? |
| `test_generator_payload_success.py` | Is a generated row marked successful only after its scientific values survive validation, writing, and exact read-back? |
| `test_generator_publication_bridge.py` | Does the real generator keep new files private until every required file is ready, closed, and safe to publish together? |
| `test_generator_run_control.py` | Do the three legal run choices select new work, resume, or append without allowing one output kind to overwrite another? |
| `test_grid2d_staging_row_contract.py` | Do in-memory and disk-backed Grid2D inputs select the same scientific rows in the same order? |
| `test_parameter_table.py` | Are sampled and derived parameter columns selected by name instead of by a remembered column number? |
| `test_mps_generator_dark_energy_binding.py` | Does every generated Syren base row reuse the explicit dark-energy law obtained once during setup? |
| `test_syren_dark_energy_coordinates.py` | Do the several accepted names for the two dark-energy coordinates resolve to one consistent pair without silently replacing a missing evolution value with zero? |

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

#### Publishing one completed CMB covariance

`test_cmb_covariance_publication.py` checks the final file-writing part of the
CMB covariance command. It uses tiny NumPy archives and does not run CAMB or
train an emulator. Run it from the repository root with:

```bash
python3 -m unittest ai.tests.test_cmb_covariance_publication
```

- **Example used:** the candidate archive contains multipoles `2, 3, 4`, three
  temperature uncertainties, and a short record of how it was made. An earlier
  archive and a file created late by a second writer use visibly different
  values, so the test can tell which bytes survived.
- **What the test does:** it completes one normal publication, then separately
  stops the private write and final-name step. Another case creates a competing
  archive just before the final name is claimed. The command-line
  case places an archive at the requested name while naming a missing YAML file.
- **Pass means:** a normal run publishes the exact member names, shapes, data
  types, and values. Every stopped run removes its private temporary file. An
  archive that already owns the requested name remains byte-for-byte unchanged,
  and the command refuses it before trying to read YAML or start CAMB.
- **A refusal it proves:** neither an ordinary rerun nor a file that appears
  while the calculation is running may be replaced. A write or final-name
  failure may not leave a partial archive at the name used by readers.
- **Why it matters:** calculating a CMB covariance can be expensive. A rerun or
  interrupted save must not destroy the preceding usable scientific result.

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

#### Finding a dataset from the filename written in a YAML file

`test_dataset_locator.py` checks the small read-only record that connects a
familiar chain filename to the generated dataset that currently owns it. This
record is called a **locator**. It does not contain a particular generation
name, so the same locator can find a newer accepted generation later.

- **Example used:** the logical filename is `params.1.txt`. The test publishes
  a first generation and then a second one. The locator file stays unchanged,
  while a new lookup finds the second generation and an earlier reader keeps
  its valid view of the first.
- **What the test does:** it installs the locator twice, loads it by basename
  and by its full path inside `chains/`, and tries to change its scientific
  request, file list, formatting, permissions, and destination.
- **Pass means:** repeated installation leaves the same read-only bytes in
  place. The locator accepts only the exact logical filename, request, output
  list, and dataset folder that created it.
- **A refusal it proves:** a different random seed cannot take over an existing
  logical filename. A writable, reformatted, linked, missing, nested, or
  parent-traversing locator also stops before any generated file is returned.
- **If this module fails:** a YAML filename may find the wrong calculation, or
  a later publication may become unreachable. Do not train from that filename
  until the locator failure is understood.
- **Why it matters:** users should not have to edit a training YAML every time
  a complete replacement generation is accepted, but one stable name must
  never silently change scientific meaning.

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

#### Moving generator output from private work to an accepted generation

`test_generator_publication_bridge.py` checks the production code that joins
the numerical generator to immutable dataset publication. The test extracts
only those real methods, so it does not start Cobaya, MPI, or a scientific
calculation.

- **Example used:** a new chain-only run writes all required files inside one
  private folder. A resume copies an accepted generation into a different
  writable folder. The examples also use one memory-mapped array and a family
  represented by a dictionary of memory-mapped arrays.
- **What the test does:** it checks the request fingerprint made from parsed
  YAML and the final uniform-sampling bounds, publishes a fresh draft, resumes
  from an unchanged accepted generation, and inspects the constructor order
  used around MPI worker startup and shutdown.
- **Pass means:** no familiar flat output file appears while work is in
  progress. Memory-mapped files are flushed and closed, all MPI workers have
  stopped, the failure mask contains only successful rows, and then the whole
  generation becomes visible in one step.
- **A refusal it proves:** a second fresh run cannot replace an existing
  accepted generation. A crash before the first publication is not presented
  as resumable work. Append stops after checking the existing generation
  because the random-number and sampler state required for an exact append is
  not yet saved. A failure-mask row containing `1` stops publication.
- **If this module fails:** the generator may expose a partial file group,
  replace newer work, publish a failed scientific row, or continue sampling
  from incomplete state. Keep the preceding accepted generation in use.
- **Why it matters:** the lower-level publication tests prove the file
  mechanism. This module proves that the real generator calls that mechanism
  at the correct points.

#### Resolving training and validation files from a user YAML

`test_cocoa_dataset_resolution.py` checks the step that reads the filenames in
a training YAML and replaces them with files from complete accepted
generations. Training and validation are looked up separately, so each one is
fixed to one generation before data loading begins.

- **Example used:** temporary datasets cover a CosmoLike vector, one CMB
  spectrum with its multipole values, background quantities with redshift
  values, and matter-power surfaces with redshift, wavenumber, and optional
  Syren base files.
- **What the test does:** it writes a small user YAML, resolves its logical
  filenames, and compares every resulting path with the exact member of the
  selected training or validation generation. It also records the generation,
  member fingerprints, and scientific request in ordinary YAML-safe values.
- **Pass means:** a payload, parameter table, covariance, failure mask,
  coordinate array, and optional base all come from the correct selected
  generation. Chain-only scalar data add no payload or failure-mask path.
- **A refusal it proves:** loose older files are not accepted when no locator
  exists. A user YAML cannot insert the resolver's private source record.
  Different probes, parameter order, scientific facts, coordinate bytes, or a
  payload borrowed from the other dataset stop before staging.
- **If this module fails:** training may combine parameter rows from one
  generation with vectors or coordinates from another. The resolved YAML must
  not be used until every related path points to its proper generation.
- **Why it matters:** individually valid files are not enough. The files used
  for one training source must describe the same rows and the same science.

#### Keeping failed generated rows out of training

`test_failed_row_staging.py` checks how the saved failure mask controls row
selection. In that mask, `0` means the generator completed the row and `1`
means it did not produce a usable scientific vector.

- **Example used:** four parameter rows have mask values `0, 1, 0, 1`. Only
  original rows 0 and 2 may enter the pool from which a fixed random seed
  chooses training rows.
- **What the test does:** it loads aligned parameter and payload files, applies
  the mask before seeded selection, reports the remaining pool size, and saves
  a fingerprint of the exact original row order used for training.
- **Pass means:** parameters and payloads stay aligned on rows 0 and 2, the
  available pool is two rows, and reversing the selected order changes the
  saved row-order fingerprint. Scalar chain-only loading still records its
  original disk rows even though it has no data-vector failure mask.
- **A refusal it proves:** requesting three usable rows stops because only two
  succeeded. A missing mask for a full dataset, the token `true` instead of
  literal `0` or `1`, a short mask, a saved selection that omits a loader row,
  or an equal-length selection in the wrong row order also stops.
- **If this module fails:** a zero placeholder written after a failed
  scientific calculation may be treated as a real training target, or the
  saved training record may describe a different row order from the one used.
- **Why it matters:** a failed generator row can have a legal shape and finite
  zeros. The separate mask is what distinguishes that placeholder from a
  valid scientific result.

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

#### Generator settings checked before output creation

`test_generator_ingress.py` checks the information a generator receives from
its YAML, covariance file, command options, and sampled rows. It runs on the
CPU with small temporary inputs. It does not call CAMB or CosmoLike and does
not train an emulator. Run it from the repository root with:

```bash
python3 -m unittest ai.tests.test_generator_ingress
```

- **Example used:** a valid request writes `ord` as one outer list containing
  `[H0, omegam]`. Its fiducial mapping supplies finite values for both names.
  A covariance header adds one unused `nuisance` name and supplies the matching
  three-by-three matrix, then requests the two sampled rows in `ord` order.
- **What the test does:** it calls the small validation functions used by the
  generator. These functions check the exact `train_args` fields, ordinary
  YAML number and Boolean types, parameter-to-covariance alignment, fiducial
  conversion, prior-bound conversion, supporting filenames, display labels,
  and unique sampled rows at the saved parameter precision.
  Production-source checks also prove that each shared command option still
  calls its named strict validator. The row-count example passes four tiny
  arrays directly to the selector; it does not run MCMC.
- **Pass means:** valid native YAML integers, numbers, and Booleans keep their
  meaning; the requested covariance is selected in `ord` order; the final row
  count equals the requested count; and a parameter without usable `latex`
  text uses its parameter name as the display label.
- **A refusal it proves:** a duplicate or wrongly nested `ord`, a sampled name
  containing whitespace, a covariance filename that leaves the YAML folder,
  a repeated covariance header, a nonsquare or nonfinite matrix, meaningfully
  different opposite covariance entries, a nonpositive variance, a Boolean
  fiducial, a quoted point count, a control character in a display label, an
  unknown `train_args` setting, or too few unique MCMC rows is rejected by the
  validator that runs before output creation.
- **Why it matters:** a malformed request can still contain plausible names
  and numbers. Refusing it before file creation avoids leaving a smaller or
  differently ordered dataset that looks complete.

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

#### Matching a parallel result to its assigned row

`test_generator_mpi_message_binding.py` checks the messages returned by
parallel data-generator workers. MPI is the tool that lets rank zero give
different parameter rows to several worker processes at the same time.

- **Example used:** rank zero assigns row 4 to worker 2 and row 17 to worker 3.
  Worker 3 replies first, followed by worker 2. A stale, repeated, or damaged
  reply may instead name row 5, arrive from a worker with no current
  assignment, or claim that the wrong worker finished shutting down.
- **What the test does:** it runs the real message validators without starting
  MPI or a scientific calculation. It also reads the generator source to
  confirm that both result-receiving loops and the shutdown loop validate a
  message before removing the worker's assignment.
- **Pass means:** a valid result keeps its payload and uses the row stored in
  rank zero's assignment. A valid shutdown reply names the same worker that
  sent it. The validator itself never edits the assignment table.
- **A refusal it proves:** an unknown worker, a duplicate reply, a different
  row number, a Boolean or negative row, an unknown result kind, a malformed
  error report, or a false shutdown reply stops before a data-vector row is
  changed.
- **Why it matters:** without this binding, a perfectly finite scientific
  vector calculated for one cosmology could be written beside another
  cosmology's parameters. The saved arrays would have valid shapes and could
  silently train the emulator on false row-to-target pairs.

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

#### Resolving dark-energy coordinate names

`test_syren_dark_energy_coordinates.py` checks the shared conversion from
`w`, `w0`, `wa`, and `w0pwa` to the two numbers read by the Syren analytic
matter-power formula. Here `w` and `w0` are two names for the present-day
equation of state, while `w0pwa` means `w0 + wa`.

- **Example used:** one point gives `w0 = -0.8` and `wa = 0.3`. An equivalent
  point gives `w = -0.8` and `w0pwa = -0.5`, from which the resolver must
  recover the same nonzero `wa`.
- **What the test does:** it tries both complete coordinate forms, all four
  names together, and the explicit time-varying, constant-`w`, and
  cosmological-constant laws. It also sends Python and NumPy real scalars and
  values just inside and outside the one tolerance allowed for numbers stored
  in `float32` data.
- **Pass means:** equivalent forms produce the same `(w0, wa)` pair. Every
  redundant value agrees with relative tolerance zero and absolute tolerance
  `4 * numpy.finfo(numpy.float32).eps`. `syren_params_from` still returns the seven
  numbers expected by the generator and adapter.
- **A refusal it proves:** conflicting `w` and `w0`, an inconsistent
  `w0pwa`, Boolean, text, array, complex or nonfinite input, an unknown law,
  or a lone present-day value with no explicit constant-`w` law must stop.
- **Why it matters:** setting a missing `wa` to zero produces a smooth finite
  matter-power spectrum for the wrong cosmology. Shape and finiteness checks
  cannot detect that scientific substitution later.

#### Keeping one dark-energy law from generation to serving

Four additional modules check where those resolved coordinates enter and
leave the saved matter-power workflow.

- `test_generator_dark_energy_facts.py` gives the dataset-description code the same
  parameter pattern as the shipped EMUL2 YAML: `w0pwa` and `w` are sampled,
  while `wa` is calculated from them. Pass means the saved law is
  time-varying `w0wa-cpl`, with physical inputs `[w, wa]`; it is never mislabeled
  as constant `w` merely because `wa` was not sampled directly.
- `test_mps_generator_dark_energy_binding.py` builds tiny positive power
  surfaces in memory. It proves that setup asks for the law once, reuses it
  for every row, and that a
  nonzero `wa=0.2` reaches both Syren base formulas. Separate examples show
  that zero may be supplied only by the explicit constant-`w` or
  cosmological-constant law.
- `test_mps_dark_energy_adapter.py` checks the other side of the saved
  emulator. A dropped `w0pwa` is removed from the names requested from Cobaya;
  calculated `wa` is requested instead. The adapter then reconstructs all
  names needed by a saved predictor before either learned model or analytic
  base runs. A saved file that lists `w0pwa` as an input but calls its law
  constant-`w` is refused with instructions to regenerate it.
- `test_dark_energy_vertical_identity.py` checks the narrow fixed-value
  comparison used at serving time. It compares concrete values only when the
  artifact and live model use the same coordinate name. Renamed, derived, or
  transformed coordinates remain the responsibility of the custom Cobaya
  parameterization.

`test_mps_dark_energy_real_cobaya.py` joins these pieces with Cobaya's real
parameter engine. It samples `w0pwa=-0.7` and `w=-0.9`, so the calculated value
is `wa=0.2`. Pass means Cobaya sends only `w` and `wa` to the Theory, the
data generator saves the same physical law, and the data generator and Cobaya
adapter obtain the same Syren pair `(-0.9, 0.2)`. The normal CPU suite skips
this module when Cobaya is unavailable.

To run the real check, open a terminal at the repository's top folder and
activate an environment that contains Cobaya. The command changes no project
files. It runs one test and ends with `OK`:

```bash
python -m unittest -v ai.tests.test_mps_dark_energy_real_cobaya
```

### Training calculations

| File | Question answered |
| --- | --- |
| `test_active_model_validation.py` | Does startup refuse ambiguous or unusable settings before it creates a learned model, while valid CNN and Transformer heads still receive a first training update? |
| `test_batching_sizing.py` | Does the batch planner count the actual memory used by every stored target value before training starts? |
| `test_d5_training_behavior_witnesses.py` | Do small numerical examples support the learning-rate, moving-average, activation, and frozen-layer results required by the longer training gate? |
| `test_finetune_post_step_and_provenance.py` | Does fine-tuning update one weight in the required order and save which earlier emulator supplied the starting weights? |
| `test_padded_head_identity.py` | Do CNN and Transformer heads keep storage-only rectangle cells from changing physical outputs, even after biases, activations, FiLM shifts, attention, or several head blocks? |
| `test_training_pass_recipe.py` | Does the record prepared for saving describe every training pass that actually ran, including phase-specific settings and the exact section of the loss history produced by that pass? |
| `test_trf_token_width.py` | Does a Transformer refuse a one-number token that cannot respond to its input while continuing to accept supported two-number tokens? |
| `test_warmstart_perturbed_finite.py` | Does a warm start report the exact input or output that first becomes `NaN` or infinite? |

#### Model settings checked before construction

`test_active_model_validation.py` checks values in the selected `model` block
before PyTorch creates the model's learned layers. It also constructs tiny
valid CNN and Transformer heads so refusal cases cannot replace the positive
behavior the library needs.

- **Example used:** one setting uses the Boolean `film: false`; another uses
  the quoted text `film: "false"`. Other examples include zero head blocks, a
  fractional token count, a correction gate so small that 32-bit storage
  rounds it to zero, and three attention heads acting on a token width of
  four.
- **What the test does:** it calls the shared setting validator, checks the
  public constructors under ordinary and optimized Python, counts the CNN or
  Transformer blocks that a valid request creates, and performs one
  frozen-trunk optimizer step on each valid head.
- **Pass means:** real Booleans and exact positive counts keep their stated
  meaning; an unused alternative head block does not affect the selected
  architecture; physical CNN groups and Transformer widths agree with the
  output layout; and both valid correction heads receive a finite nonzero
  gradient and change a head weight on the first step.
- **A refusal it proves:** quoted Booleans, strings or fractions in place of
  counts, zero head depth, even CNN kernels, incompatible attention widths,
  zero or nonfinite correction gates, and an activation that blocks the
  zero-initialized head all stop before a learned linear layer is allocated.
- **Why it matters:** silently converting one of these values can enable the
  opposite switch, omit a requested correction, or build a head that cannot
  begin learning. The run could otherwise finish with legal array shapes but
  the wrong model behavior.

#### Physical coordinates inside padded CNN and Transformer heads

`test_padded_head_identity.py` checks the rectangular workspace used when
physical output groups have different lengths. Extra cells make the rows the
same width, but those cells do not represent measurements.

- **Example used:** one row keeps angular positions zero and two, while a
  second row keeps positions one and two. Another example masks an entire
  middle row. Equal survivor counts therefore cannot stand in for the actual
  coordinate map.
- **What the test does:** it routes values through real one- and two-block CNN
  heads, real Transformer attention and MLP layers, an activation that maps
  zero to one, and a FiLM shift. It also changes only an artificial value to a
  large positive or negative sentinel and compares every physical result.
- **Pass means:** original angular positions remain distinct; a fully masked
  row stays finite and exactly zero without shifting the following row; and a
  complete rectangular head keeps the former live convolution calculation
  exactly.
- **A refusal it proves:** an older geometry containing only per-row counts is
  rejected because those counts cannot recover which angular positions were
  kept.
- **Why it matters:** an artificial cell can otherwise carry a value through a
  later block and change a valid scientific output while all tensor shapes
  still look correct.

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

#### Recording every training pass

`test_training_pass_recipe.py` checks the training record that
`run_emulator` prepares for the saved `.h5` file. A **pass** is one continuous
part of training with one set of settings. A run may use one pass for the
whole model, separate passes for the trunk and head, or an additional
transfer-refinement pass.

- **Example used:** ten artificial training rows are read eight at a time. A
  five-epoch run first trains the trunk for two epochs and then trains the
  joined model for three epochs. The trunk uses learning rate `0.008`, while
  the later pass uses `0.004`. A second example runs two ordinary epochs and
  one transfer-refinement epoch.
- **What the test does:** it replaces the expensive numerical update with a
  deterministic CPU stand-in, but runs the real pass-planning code. It then
  inspects the record returned by that code. No emulator file is written and
  no scientific data are loaded.
- **Pass means:** each executed pass has its role, epoch count, learning rate,
  loss, scheduler, warmup, clipping, moving average, rewind choice, anchor,
  chunk size, steps per epoch, and applied compile mode. History intervals
  are ordered without gaps: the example records trunk history `[0, 2)` and
  later history `[2, 5)`. The transfer example records its extra epoch as a
  separate pass `[2, 3)`, rather than hiding it inside the ordinary pass.
- **A refusal it proves:** changing an override, omitting transfer refinement,
  assigning two passes the same history rows, or reporting the configured
  compile mode when a different mode was applied makes the test fail.
- **Why it matters:** a loss plot alone shows only a sequence of numbers. The
  pass record explains which training decision produced each part of that
  sequence and lets a later reader reproduce the work.

This test is different from `test_model_recipe.py` below. The
training-pass test answers **what optimization work ran**. For example, it
records that the trunk used two epochs at learning rate `0.008`. The artifact
recipe test answers **which model must be constructed to read the learned
weights**. For example, it records whether those weights belong to a ResCNN
with a three-wide kernel or to a different supported design. A complete saved
emulator needs both answers.

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

A saved emulator has two files. The `.emul` file contains its learned tensors.
The `.h5` file contains the scientific record and the instructions needed to
interpret those tensors. HDF5 divides that second file into named sections,
similar to named folders inside a file. Rebuilding means checking that the two
files belong together and constructing the model needed for new predictions.

| File | Question answered |
| --- | --- |
| `test_artifact_recipe_preflight.py` | Does saving stop before reading model weights when the model recipe is incomplete or the five history arrays have incompatible shapes? When reopening, does a damaged reconstruction input—such as the model recipe, saved geometry, composition record, or output identity—stop before saved Python classes or checkpoint tensors are used? The same test proves that removing the optional history group does not prevent reconstruction. |
| `test_model_recipe.py` | Does a model recipe name every constructor choice needed to rebuild the six supported model designs, without silently supplying a current software default? |
| `test_artifact_output_identity.py` | If two runs train different spectra, distance quantities, matter-power products, survey probes, scalar columns, model settings, selected rows, or source emulators, will they receive different output names? If only the checkout path or dictionary order changes, will the scientific name stay the same? |
| `test_artifact_transfer_state_contract.py` | Does a transfer artifact store its base weights without duplicate state hashes, and does ordinary strict model loading refuse missing, extra, or wrong-shaped tensors? |
| `test_cobaya_adapter_contracts.py` | Do all five Cobaya adapters interpret settings strictly, combine only compatible cosmic-shear sections, publish scalar results through Cobaya, and give each reader an independent result object? |
| `test_dark_energy_vertical_identity.py` | Does serving compare a concrete fixed value when the artifact and live model expose it under the same name, without interpreting custom aliases or transformations? |
| `test_grid2d_const_mask.py` | Does a saved Grid2D geometry remember exactly which coordinates use fixed stored values instead of neural-network predictions? |
| `test_mps_dark_energy_adapter.py` | Does the matter-power adapter ask Cobaya for physical `w, wa`, reconstruct saved coordinate names, and stop conflicts before either predictor runs? |
| `test_mps_dark_energy_real_cobaya.py` | Does real Cobaya turn dropped sampled `w0pwa` into a nonzero calculated `wa` and give the data generator and adapter the same Syren coordinates? |
| `test_padded_head_artifact.py` | Does a structured head refuse a model/geometry layout disagreement before saving, reopen a valid pair with the exact physical map and mask, and refuse a checkpoint that omits or replaces either fixed record? |
| `test_pce_strict_selection.py` | Does a polynomial base enter a saved emulator only after a finite leave-one-out check passes in the same number format that will be stored? |
| `test_results_artifact_pair.py` | Do the learned weights and scientific record identify each other, refuse every already-used output name without changing it, and leave a new output name empty or visibly interrupted after a failed save? |
| `test_results_composition_mode.py` | Does the result file state how its neural-network output and any saved base are combined into a physical prediction? |
| `test_results_const_mask_declaration.py` | Can a reader detect one changed Grid2D fixed-coordinate position after the result was saved? |
| `test_results_rebuild_fixed_facts_names.py` | Does reopening an emulator stop when saved input names disagree, even if the structured scientific record and its saved text copy were changed together? |
| `test_mps_sigma8_contract.py` | Does the matter-power adapter calculate conventional sigma-eight with the correct physical radius, exact redshift, and enough wavenumber coverage? |
| `test_public_prediction_validation.py` | Does every public prediction stop at the first invalid number, wrong array shape, or unsupported saved target transformation, before an adapter can publish a partial result? |
| `test_schema3_production.py` | Does training stop early when a dataset has no scientific record, and does a complete current-format save reopen successfully? |

#### Complete instructions for rebuilding a model

`test_model_recipe.py` checks the model recipe used when learned
weights are reopened. The recipe contains the model class and exact
constructor choices. The saved parameter and output geometries separately
record the coordinate conversions, including an analytic target law when the
output uses one. The saved composition mode says whether the neural output is
used alone or combined with a polynomial or transfer base.

- **Example used:** the test constructs complete recipes for ResMLP, ResCNN,
  and ResTRF models and for their three intrinsic-alignment counterparts. A
  ResCNN example explicitly states its kernel width, groups, block counts,
  activation, normalization, optional head activation, and geometry choice.
- **What the test does:** it removes required fields, adds unknown fields, and
  tries unknown model, activation, and normalization names. It also confirms
  that these plain saved descriptions are checked before the named model
  module is imported.
- **Pass means:** all six supported designs accept a complete recipe. An
  explicit `null` head activation remains different from a missing field. The
  live model's own recipe must equal the saved recipe, and its tensors must
  load with exactly the expected names and shapes.
- **A refusal it proves:** a missing kernel width, an extra future setting, or
  an unknown model or factory name stops before reconstruction chooses a
  default. Numerical ranges are checked later by the trusted constructor or
  factory that owns each rule.
- **Why it matters:** software defaults can change after an emulator is
  trained. If a saved recipe omits one constructor choice, the same weights
  may later be placed in a model with a different meaning even though every
  tensor still has an acceptable shape.

Unlike `test_training_pass_recipe.py`, this test does not ask how many epochs
ran or which learning rate was used. It asks whether the learned tensors can
be placed back into the exact model that gave them meaning. Keeping these as
separate focused tests makes a failure specific: one points to the history of
training decisions, while the other points to reconstruction instructions.

#### Checking saved instructions before executable work begins

`test_artifact_recipe_preflight.py` checks the order used when an emulator is
saved or reopened. Here, **preflight** means reading and checking the plain
saved descriptions before the library may inspect model weights, import a
saved Python class, construct a geometry, or load checkpoint tensors.

- **Example used:** a complete one-epoch ResMLP artifact is saved. One copy
  loses a required model field. Another receives a model recipe that no longer
  agrees with its saved output identity. A third copy loses the entire optional
  history group.
- **What the test does:** it places sentinels on weight access, dynamic Python
  import, and checkpoint loading. Each damaged file must report its metadata
  problem while every sentinel remains untouched.
- **Pass means:** saving refuses incomplete main and transfer recipes before
  staging a file or reading `model.state_dict`. It also refuses nonfinite or
  incompatible training-history arrays. Reopening checks the model recipe,
  saved geometry and analytic law, composition mode, and output identity. It
  does not interpret the pass description or history curves.
- **A refusal it proves:** a missing constructor field, changed valid recipe,
  malformed saved geometry, or incompatible history arrays cannot be hidden by
  a checkpoint that still has plausible tensor shapes. Removing the history
  after publication does not prevent the learned model from reopening because
  those curves are a record for readers, not reconstruction instructions.
- **Why it matters:** imported code and learned tensors can perform work. A
  damaged plain record must be rejected before either surface can influence
  how the file is interpreted.

This integration test joins the two narrower checks above without mixing their
jobs. The model-recipe test defines the fields needed to rebuild the network.
The training-pass test explains how training produced the weights. The
preflight test proves that reconstruction checks its actual inputs early, while
leaving the historical training record out of the prediction path.

#### Loading the saved transfer model strictly

`test_artifact_transfer_state_contract.py` checks the base-model tensors
stored inside a transfer artifact. A frozen run stores one base state. A
refined run also stores `drifted_state`, and the separate transfer lifecycle
gate proves that prediction selects those refined weights.

- **Example used:** the test saves a tiny transfer model, then makes three
  separate damaged copies. One copy loses a tensor, one gains an unexpected
  tensor, and one changes a tensor's shape.
- **What the test does:** it reopens each copy through the public reader and
  reaches PyTorch's ordinary strict state loading. It also confirms that a new
  transfer file contains no duplicate state-digest attributes or configuration
  fields.
- **Pass means:** the complete saved state rebuilds normally, while each
  damaged tensor layout is refused by the model that must consume it.
- **A refusal it proves:** missing, extra, and wrong-shaped tensors cannot be
  placed into the registered base model. A same-shaped value edit is not
  detected by this check and remains the user's responsibility.
- **Why it matters:** one direct loading rule is easier to audit than hashing
  the same embedded tensors several times and copying those hashes into the
  same result file.

The remaining tests in this file cover target scaling. Schema 3 refuses a
transformed target mode because public prediction cannot invert it;
`rescale: none` is the valid control.

#### Scientific output names distinguish completed runs

`test_artifact_output_identity.py` checks the name appended to a trained
emulator and to its diagnostic PDF. The readable part names the family and
product, such as `cmb-tt`. A 32-character digest after that prefix summarizes
the completed run. This prevents two scientifically different results from
silently receiving the same filename.

- **Examples used:** one case compares CMB `TT` with `EE`. Other cases compare
  Hubble with transverse distance, linear matter power with nonlinear boost,
  cosmic shear with galaxy-galaxy lensing and galaxy clustering, and scalar
  outputs in two different column orders. A CMB case keeps `TT` fixed and
  changes the covariance arrays that define its multipoles, whitening scale,
  and fiducial spectrum. The tests also change an activation, a hidden width,
  a train or validation generation, a selected-row order, a polynomial base,
  a fine-tune source, and a transfer source.
- **What the test does:** each case builds two small records in memory. It
  compares their full digests and filename tags. No model is trained and no
  scientific table is read. Separate cases move the same authenticated files
  to a different folder and reverse dictionary insertion order.
- **Pass means:** every scientific change receives a different digest. Moving
  identical authenticated inputs or reordering mapping keys leaves the digest
  unchanged. Both public training drivers use the same completed identity for
  the saved emulator and diagnostic.
- **A refusal it proves:** production naming stops if either train or
  validation lacks its published generation, exact staged-row order, member
  digests, or source-emulator pair identity. A pathname alone cannot stand in
  for an authenticated fine-tune or transfer source.
- **Why it matters:** equal model names and row counts do not imply equal
  science. Without the complete identity, training `TT` after `EE`, for
  example, could overwrite a valid emulator with a different result.

#### Polynomial-base accuracy before saving

`test_pce_strict_selection.py` checks the polynomial-chaos base used by an
NPCE emulator. NPCE combines this fixed polynomial prediction with a neural
correction. The polynomial part must pass its own accuracy limit before the
neural model is trained or a result is saved.

A leave-one-out check asks the fitted polynomial to predict one training row
as though that row had been excluded from fitting. Repeating this for every
row estimates how the polynomial performs on values it did not use to choose
its coefficients.

- **Example used:** one example contains a straight line and a quadratic curve
  that a degree-two polynomial can reproduce. A second contains the quadratic
  curve `x²`, but permits only a constant and a straight line. Its best
  leave-one-out error is about `1.19008`, which is easy to compare with the
  requested limit `0.000001`.
- **What the test does:** it fits the passing example, saves a real `.emul` and
  `.h5` pair, rebuilds that pair, and compares every saved polynomial array and
  prediction exactly. It then attempts the failing example inside an empty
  temporary folder. Separate cases exhaust one- and two-column polynomial
  bases, inject invalid term-selection results, and place the accuracy limit
  between the score before and after the coefficient is rounded for saving. A
  large-coefficient example also proves that cancellation hidden by a 64-bit
  multiplication remains visible in the stored 32-bit multiplication. A
  two-output example checks both columns in the same matrix multiplication,
  because computing each column alone can round differently. If one column
  then misses the limit, the fit removes that output pattern, repeats the
  joint check, and keeps a different pattern that still passes.
- **Pass means:** every retained output pattern has a finite error strictly
  below `loo_max` with the saved bounds, coefficients, and multiplication in
  `float32`, the stored 32-bit number format. A passing model rebuilds with
  identical saved polynomial arrays and predictions. The training-size sweep
  gate also recognizes exactly one finite fraction for each requested size.
- **A refusal it proves:** if every output pattern misses the limit, fitting
  stops and the temporary folder remains empty. `NaN`, positive or negative
  infinity, duplicate polynomial terms, a fit that would use all rows as
  coefficients and leave no independent row to check, a rounded coefficient
  that crosses the limit, or a sweep result equal to `NaN`, the marker for a
  calculation that did not produce a number, must not look successful.
- **Why it matters:** a finite prediction is not evidence that the polynomial
  approximation met the accuracy requested in the YAML. Saving a rejected
  base would give the neural correction and later scientific analysis a
  starting calculation that the run had already shown was too inaccurate.

#### Scientific records required before training and saving

`test_schema3_production.py` uses tiny train and validation records whose one
sampled input is named `p0`.

- **Example used:** both records declare `p0`, and a small saved emulator also
  carries the model instructions needed to rebuild it.
- **Pass means:** the records are checked before a device or earlier model is
  opened. Their exact original text reaches staging. The saved file states
  schema 3 and rebuilds on the CPU.
- **A refusal it proves:** a missing record, malformed YAML, a different input
  name, raw bytes instead of decoded text, or a missing model instruction
  stops before model weights or output files are created.
- **Why it matters:** a long training run must not finish with files that the
  library's own reader immediately rejects.

#### Learned weights matched to their scientific record

`test_results_artifact_pair.py` checks the two files that together make one
saved emulator. The `.emul` file contains learned tensors. The `.h5` file
contains the scientific facts, model instructions, and a SHA-256 fingerprint
of those exact tensor-file bytes.

- **Example used:** two tiny models have the same architecture and tensor
  shapes but different learned values. Each is saved with its own record.
- **What the test does:** it rebuilds an unchanged pair, then copies the second
  model's tensor file beside the first model's record. It tries to save over a
  complete pair, either lone member, either symbolic-link member, and an
  interrupted-save marker. It also interrupts a fresh save between the two
  final filename changes and injects ordinary HDF5, allocation, and rename
  failures.
- **Pass means:** the unchanged pair rebuilds. Every occupied name refuses
  before temporary-file creation and remains byte-for-byte unchanged. An
  ordinary failure on a new name removes the new partial files. A hard
  interruption leaves a visible marker that makes the incomplete root refuse.
  If two writers race for one unused name, the first completed pair wins and
  the later writer cannot replace it. Warm-start receives its model
  instructions and data settings from the same authenticated HDF5 open rather
  than reopening a pathname that another process could change.
- **A refusal it proves:** swapped tensors stop before PyTorch or model
  construction begins. Missing or malformed identifiers stop. A checkpoint
  containing a text value or an unsafe pickle operation is opened with
  `weights_only=True`, performs no side effect, and is refused before a model
  is constructed.
- **Why it matters:** matching tensor names and shapes cannot prove that
  weights were trained under the scientific assumptions in the neighboring
  record. Loading unrestricted pickle data can also run code instead of merely
  reading model tensors.

#### Padded-head coordinates preserved across save and rebuild

`test_padded_head_artifact.py` saves a small CNN whose four physical outputs
occupy four named positions in a two-by-three rectangle. The other two cells
exist only so both rows have the same stored width.

- **Example used:** the coordinate map is `[0, 2, 4, 5]`, and the aligned mask
  is `[[true, false, true], [false, true, true]]`. The final convolution has
  deterministic nonzero weights, so the head changes the trunk prediction.
- **What the test does:** it saves through `save_emulator`, rebuilds through
  `rebuild_emulator`, and compares the live and reopened predictions, geometry
  records, and fixed model buffers exactly. It then asks the writer to combine
  a model with a different geometry and proves that no staging begins and a
  preceding valid pair remains unchanged. Separate saved copies remove a mask
  or replace both checkpoint buffers with another internally consistent
  layout.
- **Pass means:** the live correction survives exactly, and both saved files
  independently agree on which rectangle cells are physical. The writer also
  prevents a disagreement instead of publishing a pair that its own reader
  would reject.
- **Refusals it proves:** a live model without `pad_valid`, or one paired with
  the wrong geometry, stops before file staging. A saved checkpoint without
  `pad_valid`, or one whose map disagrees with the HDF5 geometry, stops before
  ordinary state loading.
- **Why it matters:** a valid checksum proves which two files belong together;
  it does not prove that both files assign the same scientific meaning to each
  tensor position.

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

#### Invalid values stopped during a public prediction

`test_public_prediction_validation.py` checks the path used after a saved
emulator is opened and asked for a new prediction. It also checks the final
background, scalar, and matter-power calculations performed by the Cobaya
adapters before those results become visible to the sampler.

- **Example used:** a two-parameter toy emulator produces a two-number model
  result and scatters it into a three-number public data vector. Separate
  cases place `NaN`, infinity, or the wrong array shape in the input encoding,
  model result, physical decoder, or final scatter operation.
- **What the test does:** it asks the public predictor to process each damaged
  value and requires the error to name the first failed stage. It also opens
  small saved-attribute stand-ins, exercises the CMB amplitude calculation,
  and runs background, scalar, and matter-power adapters with one deliberately
  invalid intermediate result.
- **Pass means:** ordinary Python and NumPy real numbers still predict
  normally. Every supported emulator family returns its expected public
  shape. A saved result can be served only when it explicitly records that no
  unavailable target transformation must be reversed. An adapter publishes
  nothing when one of its later calculations fails.
- **A refusal it proves:** Booleans, text, arrays in place of scalar
  parameters, nonfinite values, broadcastable but wrong matter-power shapes,
  a nonpositive Hubble rate, an invalid CMB amplitude factor, and transformed
  saved targets all stop before a partial scientific result is cached.
- **Why it matters:** a later decoder can sometimes replace or hide a bad
  model coordinate, and NumPy can silently broadcast a row across a surface.
  Checking only the final array could therefore turn an invalid intermediate
  calculation into a finite-looking but scientifically wrong result.

#### Cobaya adapter settings, assembly, and returned results

`test_cobaya_adapter_contracts.py` checks the boundary between saved
emulators and the Cobaya components that use them. It does not train a model.
Most cases use small predictors with fixed facts so that a failure identifies
the adapter rule rather than model fitting or HDF5 input.

- **Examples used:** one setting supplies the text `"false"` where a Boolean
  is required. Two path spellings point to the same saved root. Two
  cosmic-shear sections are listed in reverse order but represent distinct
  physical blocks. A first likelihood changes an array returned by the
  adapter before a second likelihood asks for the same result.
- **What the test does:** it runs the shared setting checks through all five
  public adapter initializers. It builds valid and invalid cosmic-shear
  section plans, requests an exact CMB multipole range, publishes one scalar
  result, and reads background and matter-power results twice. When the real
  Cobaya package is installed, one additional case asks a small likelihood
  for `rdrag` and follows that value through Cobaya's normal derived-result
  path.
- **Pass means:** settings keep their documented types, saved roots are
  unique, disjoint shear sections follow physical block order, scalar output
  appears under `derived`, and every second reader sees the original arrays
  and metadata. A matter-power pair that already requires `As_1e9` does not
  gain a redundant `As` input.
- **A refusal it proves:** quoted Booleans, invented devices, malformed or
  repeated roots, overlapping shear blocks, incompatible layouts, two full
  shear vectors, a wrong section width, and a non-integer CMB maximum all
  stop before they can become a scientific likelihood result.
- **Why it matters:** these mistakes can preserve valid array shapes while
  reordering or repeating physical measurements. Returning an internal array
  directly also lets one consumer silently change the value served to the
  next consumer.

#### Conventional sigma-eight from matter power

`test_mps_sigma8_contract.py` checks the derived matter-fluctuation number
called sigma-eight. Its spherical averaging radius is 8 Mpc/$h$, where
`h = H0/100`. Because this repository stores wavenumber in inverse Mpc, the
number used for the radius must be `8/h` Mpc rather than the literal number 8.

- **Example used:** the artificial spectrum
  $P(k)=512\pi^2/(9k)$ has an analytic answer. With `H0 = 64`, `h = 0.64`,
  and radius `8/h = 12.5` Mpc, the infinite-range answer is exactly 0.64.
  On the finite checked grid the independent answer is
  `0.6399980037465730`. The old 8-Mpc calculation returns approximately 1,
  so it cannot pass by rounding to the same value.
- **What the test does:** it calculates the result from both float64 and
  float32 inputs, then runs a lightweight real Cobaya model. The live model
  proves that a sigma-eight request supplies `H0` to the adapter, while an
  ordinary linear or nonlinear matter-power request still works without an
  unnecessary `H0` dependency. Three paired numerical examples also sit just
  below and just above the missing-tail, lower-resolution, and largest-panel
  limits. Each pair proves that its named check can refuse a bad grid without
  hiding behind one of the other two checks.
- **Pass means:** the adapter uses `8/h` Mpc, reads only an exact stored
  `z = 0` row, accumulates the integral in float64, and publishes the derived
  value only after all of those checks pass.
- **A refusal it proves:** `z = 0.009` is not treated as zero. A grid covering
  only low wavenumbers, only high wavenumbers, or `k = 1..10` stops because
  the measured contribution has not decayed at both edges. A grid with only
  eight widely separated values also stops because coarsening it changes the
  integral too much.
- **Why it matters:** all of these wrong calculations can return finite,
  positive numbers. Array-shape and finiteness checks alone therefore cannot
  protect a scientific analysis from a mislabeled smoothing scale or from a
  grid that omitted most of the integral.

## Three kinds of AI workflow checks

The role system separates three questions. This keeps ordinary explanatory
writing from becoming an accidental software interface.

- A **behavior test** calls working code and checks what it does. For example,
  `test_role_workflow_behavior.py` simulates a rejected Git push. It requires
  one ordinary push attempt, no force retry, and a saved reminder that the
  push is still owed.
- A **schema test** checks structured fields and allowed values.
  `test_role_contract.py` changes `landing.force_push_allowed` in a temporary
  contract and confirms that the daemon refuses the disagreement. The YAML
  field, not a sentence in a role guide, is authoritative.
- A **documentation-consistency test** checks that a reader can still find the
  canonical rule, command, or section. `test_role_directive_contract.py`
  is being narrowed to stable paths and machine examples. Its remaining prose
  checks include some legacy overlaps with behavior and schema tests. Later
  concept-sized changes should migrate those overlaps before removing them.
  The finished suite should not require one particular explanation when a
  clearer equivalent paragraph would teach the same rule.

The first two kinds decide whether the software obeys a rule. The third keeps
the explanation discoverable. When a hard rule can be tested through code or
structured data, do that instead of copying its prose into an assertion.

## AI workflow and policy test inventory

These modules check the rules used by the Architect, Implementer, Red Team,
mailbox watcher, and repository protection tools.

| File | What it checks |
| --- | --- |
| `test_backlog_guard.py` | Creates a small local backlog and records the SHA-256 fingerprint of its exact bytes, just as the Architect does before and after an edit. An unchanged backlog must pass; a changed backlog must fail until the Architect supplies the previously accepted fingerprint and saves the new one. The test also proves that an Implementer or Red Team cannot authorize that update, and that a missing file, link, hard link, directory, oversized file, malformed saved record, or file changed during reading is refused instead of being mistaken for an approved backlog. |
| `test_handoff_contract.py` | Writes sample Architect and Red Team instruction files, then passes them to the same validator used by the mailbox tools. A valid file must contain the ordered plan, exact files and commands, evidence destination, severity decision, and character limit. The Architect must either name bounded helper jobs or explain concretely why another session would only repeat the same work without independent evidence. Required helper blocks record edit permission, ownership, task, returned artifact, observable acceptance result, and stop condition. Separate cases prove that the Implementer cannot omit a required helper, invent a no-helper waiver, or change the Architect's saved reason. Other cases refuse vague work, overlapping edit ownership, fabricated capability failure, missing sections, placeholders, hidden Markdown, oversized or non-UTF-8 notes, and a different validator program. |
| `test_implementer_checkpoint_hook.py` | Calls the pause hook with chosen times before and after the Implementer's 90-minute boundary. Before the boundary the hook says nothing; at the boundary it asks for one progress handoff, and repeated or helper-agent events cannot claim a second instruction. It also checks that a revised Implementer turn receives a fresh period and that a checkpoint goes to the Architect without the ordinary landing instruction. |
| `test_mailbox_conditional_preamble.py` | Builds one prompt for a role that must send work onward and one terminal prompt for a role whose job ends there. Only the first may require a reply message; the terminal prompt must say that no reply is required and must not contain a second general instruction that contradicts that ending. |
| `test_mailbox_candidate_delivery_recovery.py` | Interrupts candidate delivery at several saved steps. Restart must preserve the exact Implementer commit, refuse globally protected files, and send an ordinary unplanned file to the Architect as `SCOPE_EXCEEDED` instead of losing or silently accepting it. |
| `test_mailbox_candidate_state_recovery.py` | Damages or partly writes the private candidate record and Git reference in controlled temporary repositories. Recovery may adopt one proved descendant candidate, but it must refuse a missing, unrelated, or conflicting identity. |
| `test_mailbox_daemon_architect_entrypoint.py` | Runs the public `send` command in a temporary mailbox. A user request must create exactly one message to the Architect with the original text and chosen severity; public commands that address the Implementer or Red Team must refuse with no new file, while the watcher may still deliver the Architect's internal messages to those roles. It also confirms that the old targeted ping spellings are rejected because the direct provider check no longer addresses a mailbox role. |
| `test_mailbox_primary_backlog_bridge.py` | Starts from an older checkout whose local backlog still lives outside the saved primary worktree. The bridge must copy one valid backlog into the primary without replacing a newer file or following a redirected path. |
| `test_mailbox_provider_ping.py` | Replaces the real Claude and Sol programs with small controlled subprocess results, so no test spends AI credits. Bare `--ping` must receive an exact unpredictable answer from both services. `--ping --skip-redteam` must never start Sol. Other examples simulate a missing program, timeout, wrong answer, and a program that merely echoes the prompt; each must fail without creating a mailbox request, changing the backlog, or hiding the other service's result. |
| `test_mailbox_daemon_severity.py` | Sends discovery requests with no severity and with `high`, `medium`, or `low`, then records which value each started role receives. Missing values become `medium`, malformed headers never launch work, and fix-only mode or a disabled Red Team remains stronger than a request to search at low severity. Other cases build Critical, High, Medium, and Low backlogs and prove that their counts control work order without changing anyone's job. In particular, Sol remains the advisory Red Team at every severity; a message that tries to assign implementation work to Sol is refused before an AI role starts. |
| `test_mailbox_clean_all.py` | Builds disposable Git repositories containing unfinished Claude work, an unmerged Sol commit, an old `worktree-agent-*` branch, a detached audit folder, and an abandoned unregistered folder. The explicit `--clean-all` command must remove all of that AI work while preserving `main`, a student branch and worktree, a tag, a remote-branch record, and a stash. A second run must remain harmless. Separate examples hold a live watcher lock or combine cleanup with `--once`; both must refuse before deleting anything. |
| `test_permanent_note_guard.py` | Copies the eleven permanent Markdown notes, the structured role contract, and their guard into a temporary Git repository. It proves that the note count remains eleven while an unstaged, staged, or committed change to either kind of protected knowledge is refused. It also refuses an extra tracked note, a note replaced by a link, a changed guard program, or a shortened starting-version identifier. |
| `test_permanent_note_style_contract.py` | Reads every permanent note as text and checks rules that make the notes useful after the current ticket is gone. The files must use neutral present-tense language, avoid dates and personal diary labels, provide unique stable link targets, and never depend on a temporary audit or backlog file. |
| `test_protected_policy_review.py` | Builds one proposed change to a protected role or permanent note. It proves that this review is separate from a ticket cycle, remains available during fix-only maintenance, and treats both role files as protected Architect-owned text rather than ordinary documentation. The role contracts separately require one advisory response and forbid a second review round. |
| `test_role_directive_contract.py` | Contains the remaining links between role guides, permanent policy notes, command examples, and reader guides while those checks are narrowed in concept-sized changes. These are documentation checks, not the authority for daemon behavior. Explanatory paragraphs may be reworded after an equivalent behavior or schema check protects the underlying rule. |
| `test_role_contract.py` | Reads the protected YAML contract with the same strict reader used by the watcher. The valid repository copy must say that a protected-policy draft receives one adversarial review, Claude role branches use `claude/`, Sol uses `codex/`, and cleanup recognizes the old `worktree-agent-*` form. Duplicate, unknown, missing, or wrongly typed fields must stop. Configurable timing and added candidate protections come directly from this file: one example adds `.ai-secrets/` only to a fixture contract and proves that candidate admission immediately protects it. Separate examples refuse removed safeguards, redirected trusted files, weakened landing authority, a changed contract during a running process, an unsupported worktree migration, and a live mailbox outside every protected prefix. |
| `test_role_workflow_behavior.py` | Calls role-boundary code directly. It checks exact Architect approval serialization, candidate file-scope outcomes, safe escaping of candidate filenames, and a rejected push that records debt without a force retry. A moved-main example requires the named integration-revalidation state to identify C, old landing L, old main M0, and new main M1 instead of asking for an unexplained full audit. No role-guide sentence can make these tests pass when the code behaves incorrectly. |
| `test_tests_readme_inventory.py` | Lists every immediate `.py` file in `ai/tests/` and extracts the backticked filenames from these inventory tables. The two sets must be identical, so adding a test without explaining it here, documenting a file that does not exist, or listing the same file twice makes the test fail. |
| `test_ticket_change_guard.py` | Creates small Git histories containing added, deleted, renamed, replaced, Unicode, binary, and non-UTF-8 files, then applies the ticket's `--max` character limit. One example saves ticket A, advances `HEAD` with ticket B, and proves that the Implementer's default check now sees A+B while the Architect's `--architect-audit --candidate FULL_COMMIT` check still measures A alone. Other cases prove that both audit flags are required together, the candidate is a full local descendant of the ticket's base, a positive limit refuses hidden, oversized, binary, or unreadable changes, zero means no size limit, and measuring does not change files already selected for the next commit. |

## Direct reproduction inventory

These scripts are not found by `unittest discover`. Each script has a
`main()` starting function, prints its own checks, and returns a nonzero exit
code when its witness fails.

| File | What it rebuilds inside temporary folders |
| --- | --- |
| `finite_contract_cuda_wording_repro.py` | Runs the finite-number gate on a machine that cannot perform its required CUDA compilation. The gate must report that this machine check is still unavailable and name the missing CUDA action; it must not print wording that could be mistaken for a scientific pass. |
| `tools_backlog_bundle_repro.py` | Creates a backlog and supporting files, packs them into `.tar.xz`, imports that archive into a new temporary folder, and compares every byte. It also tries an existing destination, changed protected files, links, names that escape the destination, malformed archives, and oversized inputs; each unsafe case must stop without replacing the receiver's files. |
| `tools_handoff_router_repro.py` | Creates a temporary Git repository and sends sample Architect and Implementer handoffs through the manual clipboard router. It checks the selected work folder, message order, lock cleanup, detailed directive, severity, and character-limit values. One set of examples plans two named helper jobs and proves that the returned `Subagent work` blocks must use those same names and order with concrete evidence; a missing, extra, renamed, reordered, or weak result is refused before an archive or local command is created. Other examples refuse a wrong role, bad source note, malformed instruction file, or any plan that turns Sol into an Implementer. Two-role notes may omit the later advisory Red Team review. |
| `tools_mailbox_daemon_dead_mailbox_repro.py` | Starts with no watcher, then with live, held, and outdated lock files, and runs preview, send, and ping commands. The user must receive the correct warning without a blocking read; a second live watcher, a required folder replaced by a link, or a pipe file that could wait forever must refuse safely. |
| `tools_mailbox_daemon_fix_only_repro.py` | Places one repair request and one request to discover a new problem in a temporary mailbox, then starts the watcher with fix-only mode. Only the repair may start, simultaneous attempts to enable the mode must agree, each role must receive the rule, two-role operation must still work, and changed copies of the daemon or disagreement between `--help` and the README must be detected. |
| `tools_mailbox_daemon_landing_debt_repro.py` | Builds a history where an exact candidate was accepted but its durable landing record was interrupted, then runs repeated watcher passes. Recovery must use that exact candidate and landing state until the parent daemon records distinct L; held locks and malformed records must not duplicate the request, and a corrected record must let later passes recover. The legacy filename does not mean that the watcher compares the separate Architect branch with `main` or creates work from a changed-line threshold. |
| `tools_mailbox_daemon_max_repro.py` | Starts tickets with an omitted, zero, positive, malformed, and conflicting `--max` value. The default must be zero, a positive value must appear in the terminal and in every role's environment, and malformed or disagreed values must stop before a role starts. |
| `tools_mailbox_daemon_no_redteam_repro.py` | Starts a watcher with `--skip-redteam` while Architect and Implementer requests are waiting. Those two roles must keep working in their separate saved Git worktrees, Red Team requests must remain waiting, and one accepted ticket must still count as exactly one cycle. Fix-only mode and simultaneous sends must stay safe, locks must be removed at the end, and a changed daemon copy must refuse. |
| `tools_mailbox_daemon_output_style_repro.py` | Runs ordinary waits, refusals, and progress reports and compares the exact terminal text with the examples in the README. The check catches unclear separator lines, unexplained all-capital messages, wrong waiting counts, or documentation that shows output the program no longer prints. |
| `tools_mailbox_daemon_primary_worktree_repro.py` | Uses temporary Git repositories to create and then reuse separate Git worktrees for the Architect, Implementer, and Red Team without letting any role edit through the user's original checkout. A worktree is a separate folder connected to the same project history. The setup cases inspect the three saved path-and-branch records, simultaneous first-time creation, interrupted creation resumed at the exact registered folder, older two-folder state refused without rewriting it, and wrong or colliding paths refused. Launch-boundary cases change a role branch or saved state immediately before and after a child starts; the child must not continue in a folder whose identity changed. The central ticket A/ticket B example freezes A under its own Git reference and detached audit folder, advances the Implementer folder to visibly different B bytes, and proves that Architect reads exactly A and Red Team later reads exact landing L. Landing cases require C and L to be distinct, preserve newer main bytes, refuse a dirty, detached, changed, or wrong-branch user checkout, and prove that a failed bounded non-force push creates durable debt without reopening or repeating the ticket. If main advances after L is prepared, the exact stale-state message names C, L, M0, and M1 while all work remains preserved. Two timed cases prove that an Implementer cannot ignore the 90-minute pause or turn its progress commit into an accepted landing. The permanent-note publisher case refuses a normal user, Implementer, Red Team, wrong primary folder, and duplicate request; only a correctly bound Architect process may queue the exact raw admin self-route. |
| `tools_mailbox_daemon_redteam_repro.py` | Sends several Red Team messages while preview mode and competing watcher processes are active. A message must be completely written before it becomes visible, only one watcher may claim it, message numbers must remain unique, and malformed bodies or unreplaced placeholder text must refuse without losing another message. |
| `tools_mailbox_daemon_rendezvous_repro.py` | Starts short-lived fake roles and observes the occasional 20-second Ctrl-C window after all work already in progress finishes. It proves that this manual stopping opportunity is separate from a ticket cycle, checks that a new request arriving during the window remains safe, exercises two simultaneous senders, and requires refusal when the backlog file cannot be understood. Finite-limit cases prove that `--cycle 1` leaves ticket B unchanged while ticket A waits for its Red Team return. A separate `--skip-redteam` case proves that `--cycle 3` reserves no more than three tickets in total and remembers those reservations after a simulated restart. |
| `tools_mailbox_daemon_role_models_repro.py` | Selects model names and aliases for the Architect and Implementer, then inspects the exact commands the watcher would launch for each role. Defaults, valid aliases, invalid names, and role routing must agree, and changing a copied daemon source file must stop the run rather than launching an unverified command. |
| `tools_mailbox_daemon_staleness_repro.py` | Starts fake role commands that finish, run out of account credits, time out, retry, or leave a request in the `inflight/` folder, which holds a message while one role is working on it. It checks finished-file order and recovery, and it requires refusal when a program file changes during the run, a required file becomes a link, a timeout record is malformed, or a copied daemon no longer matches its approved bytes. |
| `tools_mailbox_daemon_ticket_cycle_repro.py` | Gives the Architect and Implementer several messages for one temporary ticket and proves that conversation alone does not finish a cycle. Architect GO must be a five-line decision-only request bound to immutable candidate C. After that process exits, the parent daemon prepares distinct landing L, records it locally, and in normal operation waits for the Red Team review tied to that exact ticket and L. With `--skip-redteam`, the ticket finishes when L is recorded. Each ticket counts once in either case. The reproduction reserves finite-cycle slots before work starts, preserves the count across crashes, and confirms that the removed Sol-Implementer command-line option is rejected. False ticket names, changed modes, wrong role routes, reused or unrelated commits, incompatible queued messages, malformed old state, a missing backlog, and attempts to start extra work after the requested cycle count all refuse without consuming the waiting file. |

## How agents use these files

The Architect names the smallest relevant test command in the implementation
instructions. The command must distinguish the repaired behavior from the
old defect; a test that passes both versions is not enough.

The Architect first decides whether a helper can add an independent result.
If so, the Architect gives the Implementer a bounded plan for subagents.
“Bounded” means that each helper receives one small job, exact files, and a
clear stopping point. For example, one subagent can reproduce a mailbox
failure while another checks the documentation and existing tests, provided
their files do not overlap. This parallel work shortens a repair without
asking a simpler Implementer model to invent the division of work.

Each helper's plan names whether it may edit, the exact `path::symbol` it
owns, the task, what it must return, the observable acceptance result, and the
condition that makes it stop. The plan also names the Implementer as
Integrator and gives the exact final command. A sentence such as “use helpers
where useful” fails the handoff check because it does not tell a helper what
to do or tell the Integrator how to judge the return.

When the Architect requires subagents, the Implementer must use that plan and
remain the Integrator. It launches every planned helper before making an
Integrator-owned implementation edit. Independent helpers with non-overlapping
files run concurrently. The Integrator then checks every required return,
combines the pieces, runs the focused command, and then runs the full `unittest
discover` command. If the change affects a tool whose stand-alone reproduction
appears above, the Integrator runs that script separately and returns its raw
summary and exit code. When another session would only repeat the same
indivisible work, the Architect instead records a concrete no-helper reason.
The Implementer repeats that reason exactly and cannot create the waiver.

A runtime with no required subagent support must say so plainly.
On the first failed pre-edit launch, the Implementer puts the exact
`Capability checked`, `Attempted operation`, and `Raw failure` rows inside the
blocked `IMPLEMENTER_HANDOFF` under `Subagent work`. The relay fingerprints
that full handoff. The Architect can authorize a no-helper retry only by
copying those same SHA-bound rows exactly; neither role may reconstruct or
improve the wording. The Implementer must never claim that delegation happened
when it did not.

The Red Team reviews only the named change unless the user asked the Architect
for a wider search. A Red Team finding should name the test or reproduction
that demonstrates the defect. The Architect reruns required commands and
issues the final `GO` or `NO-GO`.

Tests and reproductions never update the eleven permanent notes. The
Architect handles any required permanent-note update as a separate protected
step. The parent watcher can land that clean note-only commit only when no
ordinary ticket is active. The update consumes no ticket cycle and receives
no Red Team review.

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
