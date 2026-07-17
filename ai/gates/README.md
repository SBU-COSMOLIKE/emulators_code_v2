# Gates: final checks for emulator changes

The gates board records the final checks used to decide whether a change is
ready. It covers cosmic shear, scalar quantities, cosmic microwave background
(CMB) spectra, background-distance emulators, matter-power emulators, data generation,
saved models, Cobaya adapters, fine-tuning, transfer, and the gate runner
itself. It is not limited to the cosmic-shear emulator.

The board is intended for the configured CoCoA workstation with an NVIDIA
GPU and the project data. Developers can still run the smaller checks in
[`ai/tests/`](../tests/README.md) on an ordinary CPU machine.

## Contents

**Main guide**

- [Tests, gates, and the board are different](#tests-gates-and-the-board-are-different)
- [See which gates exist](#see-which-gates-exist)
- [Check the workstation before a long run](#check-the-workstation-before-a-long-run)
- [Run one gate or the full board](#run-one-gate-or-the-full-board)
- [Read the result](#read-the-result)
- [Continue after an interruption](#continue-after-an-interruption)

**Common questions raised by developers**

*Appendices about commands and refusal*

- [FAQ A1. Which command-line options can I use?](#faq-a1-which-command-line-options-can-i-use)
- [FAQ A2. What does the workstation check inspect?](#faq-a2-what-does-the-workstation-check-inspect)
- [FAQ A3. Why did a gate refuse to start?](#faq-a3-why-did-a-gate-refuse-to-start)

*Appendices about results and files*

- [FAQ B1. When may an earlier PASS be reused?](#faq-b1-when-may-an-earlier-pass-be-reused)
- [FAQ B2. What happens after one gate fails?](#faq-b2-what-happens-after-one-gate-fails)
- [FAQ B3. Which files define and record the board?](#faq-b3-which-files-define-and-record-the-board)
- [FAQ B4. What is a golden comparison?](#faq-b4-what-is-a-golden-comparison)

## Tests, gates, and the board are different

A **test** asks one narrow question about the code. For example,
`test_missing_axis_is_a_requested_load_refusal` supplies saved CMB progress
files without `dv_ell.npy`, the file that records the multipole values. The
test confirms that loading stops before any spectrum file is opened.

From the repository root, the folder that contains `ai/` and `emulator/`, run:

```bash
python3 -m unittest \
  ai.tests.test_cmb_checkpoint_axis.CmbCheckpointAxisTests.test_missing_axis_is_a_requested_load_refusal
```

The command uses temporary files, does not update the gates board, and ends
with `Ran 1 test` followed by `OK` when the refusal works.

A **gate** is one named final check registered in `ai/gates/board.py`. A gate
may combine many tests, run a check program, start a real training job, or do
all three. For example, the `dataset-publication` gate groups focused CPU
tests into six required results. Those tests follow generated files from the
generator's private work folder to the exact saved train and validation data
that Cocoa gives to training. On the configured workstation, this command runs
that gate:

```bash
python3 ai/gates/run_board.py --gate dataset-publication
```

A successful attempt exits with code 0 and prints a line beginning
`[harness] GATE dataset-publication: PASS`. Code 0 is the number the terminal
uses for a successful command. The attempt also updates the board's saved
results and writes a text log containing the commands, output, and results.

The **board** is the ordered collection of gates plus the program that runs
them. It chooses the requested gates, checks the workstation, records one text
log for each gate that starts, and remembers which earlier results can still
be trusted.

The difference is practical:

- After changing CMB checkpoint loading, run the focused CMB test above. A
  failure points back to that loading behavior.
- When the source note requires a named gate for final review, run that gate.
  It may require several related results or a real training job.
- Run the full board before a release or after a group of related changes.

## See which gates exist

Run this from the repository root. It reads the gate list from
`ai/gates/board.py` and earlier attempt results from
`ai/gates/logs/board_status.json`; it does not run a gate or change either
file.

```bash
python3 ai/gates/run_board.py --list
```

The command prints every current gate ID, group, optional status, saved result,
and permanent note that owns the required behavior. Use this live list instead
of counting a table in this guide.

These examples show the board's range; they are not the full inventory.

| Gate ID | Concrete job |
| --- | --- |
| `artifact-output-identity` | Builds small identities for CMB, background, matter-power, CosmoLike, and scalar products. It checks that scientific changes receive different names, moving unchanged inputs does not rename them, and an existing output root is refused without changing its files. |
| `adapter-contracts` | Most checks use small stand-ins, and one check uses the installed Cobaya lifecycle; none trains a model. The gate refuses a quoted `"false"` where a Boolean is required, checks that cosmic-shear sections are assembled in physical order, checks that scalar and CMB results use the shapes Cobaya requested, and confirms that changing a returned array cannot corrupt the next result. |
| `dataset-publication` | Creates temporary generated datasets. It checks that the generator saves one complete read-only version, Cocoa selects one saved train dataset and one saved validation dataset, and rows marked as failed do not reach training. |
| `scalar-identity` | Requires a scalar emulator's prediction before saving and after rebuilding to match exactly. |
| `cmb-smoke` | Builds a small CMB dataset and covariance, trains an emulator, asks it for predictions through Cobaya, and creates diagnostic plots. |
| `bsn-smoke` | Compares a background-distance emulator with values from CAMB, the reference cosmology program used by this check. |
| `mps-smoke` | Generates and trains matter-power data, compares the Cobaya adapter with CAMB, and compares the adapter's sigma-eight integral with CAMB's independently derived value. |
| `npce-training` | Trains residual and ratio neural-polynomial examples, refuses incompatible settings, and requires finite results for both requested training sizes in its sweep. |
| `save-rebuild-drift` | Confirms that saved emulator variants rebuild with the same output after code defaults change. |

## Check the workstation before a long run

Start CoCoA by following the
[official CoCoA instructions](https://github.com/CosmoLike/cocoa/blob/main/README.md),
then change to this repository root. The shipped `ai/gates/board_config.json`
reads the CoCoA root from `$ROOTDIR` and already names the usual driver and
YAML folders. Leave it unchanged when those paths match the installation.
Edit it when CoCoA stores those folders elsewhere.

Then run:

```bash
python3 ai/gates/run_board.py --check
```

This command checks the saved Git history, unsaved files, required Python
imports, GPU availability, CoCoA root, driver folder, and YAML folder. It does
not start a gate, spend GPU time, update a board result, or write an attempt
log. Success ends with `== preflight PASSED ==` and returns code 0, the
terminal's success value. Each refusal names the failed condition so it can be
corrected before a long run.

The workstation check is global. Even a selected gate whose own calculation
uses only the CPU must pass the board's GPU, CoCoA, and path checks when it is
started through `run_board.py`. A developer who only needs the CPU test should
run that test directly from `ai/tests/`.

## Run one gate or the full board

First print a plan. A dry run shows commands but does not perform the
workstation check, run a gate, or update the saved results.

```bash
python3 ai/gates/run_board.py --dry-run --gate dataset-publication
```

The first line is `selected 1 gate(s): dataset-publication`. The next line
begins `dry-run plan`, followed by the check program the gate would start. No
board result or log is written.

On the configured workstation, run that one gate with:

```bash
python3 ai/gates/run_board.py --gate dataset-publication
```

This real run updates the board result files and writes a log for the completed
attempt. The result files are `ai/gates/logs/BOARD.md` and
`ai/gates/logs/board_status.json`. A successful run prints the PASS line shown
in the test-and-gate comparison above.

Run every nonoptional gate in board order with:

```bash
python3 ai/gates/run_board.py
```

The full board can generate data, train small models, and use the GPU. It
writes the output paths configured in `board_config.json` as well as the board
results and attempt logs.

Success exits with code 0 and prints `board run complete` with zero failed and
zero dependency-skipped gates. The exact number of gates can change; use
`--list` for the current inventory.

The optional `triangle-shading` gate runs only when named explicitly. The live
`--list` output identifies optional gates if that set changes.

## Read the result

A completed gate attempt writes a file such as
`dataset-publication.20260716-143012-123456.log` inside the existing
`ai/gates/logs/` folder. The name contains the gate ID followed by the date,
time, and microseconds, so each attempt has a different name. The log contains
the command that started the check, the program output, the individual
required results, and the final gate result.

If the board process stops before a gate finishes, its unfinished log keeps
the same name with `.inprogress` added at the end. A gate reused from an
earlier current PASS does not create a new attempt log.

After a real run, `BOARD.md` inside that folder gives one row per gate. Read it
to find the failed or out-of-date item. If its row names a completed log, open
that text file next. The runner also creates `board_status.json` there so the
program can remember earlier results.

The result column can say more than PASS or FAIL:

| Result | Meaning and next action |
| --- | --- |
| `PASS` | Every result that ran passed, at least one result ran, and any result that could not run is named as `UNAVAILABLE`. The saved files must also still match this attempt. |
| `FAIL` | The gate did not prove its required behavior. Open the log named in the row. If no log is named, read the row's Detail cell and the terminal output; a missing declared input can stop a gate before its log is created. |
| `SKIP-DEP` | An earlier gate required by this gate does not have a current PASS. It may have failed, been interrupted, never run, or become out of date. Resolve that earlier result first. |
| `interrupted` | A previous attempt started but did not finish; run the same selection again. |
| `stale-code`, `stale-input`, `stale-log`, or `stale-dependency` | Program files, input files, the saved log, or an earlier required gate changed; rerun the gate. |
| `pre-manifest` | An old PASS was saved before the gate began recording its complete file list. Rerun the gate so the new record covers those files. |
| `not run` | No saved attempt exists for this gate. |

`UNAVAILABLE` is additional information inside a PASS row, not a separate
board result. It means the gate declared a result but could not perform that
part on this run, for the reason printed beside it. The Architect must read
those reasons before issuing the final GO decision; a displayed PASS does not
turn an unavailable comparison into evidence that it succeeded.

## Continue after an interruption

The runner saves `RUNNING` before it starts a gate. If the process stops, the
next ordinary run repeats that interrupted gate. Earlier PASS gates are
skipped only when their program files, inputs, saved log, and required earlier
gates are unchanged.

For example, if a full-board run stops during `cmb-smoke`, run the ordinary
full-board command again:

```bash
python3 ai/gates/run_board.py
```

Do not add `--force-rerun-all` when the goal is to continue. That option
deliberately repeats every selected PASS gate.

## Common questions raised by developers

### Appendices about commands and refusal

#### FAQ A1. Which command-line options can I use?

The normal choices are:

| Option | What it selects or changes |
| --- | --- |
| `-h`, `--help` | Print the command summary and exit. |
| `--list` | Print the live gate list and saved states; run nothing. |
| `--check` | Check the configured workstation; run no gate. |
| `--dry-run` | Print the commands for the selected gates; run nothing. |
| `--gate ID [ID ...]` | Select only the named gates. Optional gates are allowed here. |
| `--tier backlog` | Select the nonoptional gates in the `backlog` group. |
| `--tier new-features` | Select the nonoptional gates in the `new-features` group. |
| `--tier save-and-sample` | Select the nonoptional gates in the `save-and-sample` group. |
| `--from ID` | Start at one gate and continue in board order, skipping later optional gates. |
| `--force-rerun ID [ID ...]` | Repeat named selected gates even when their saved PASS is current. |
| `--force-rerun-all` | Ask every selected gate to run again instead of reusing a current PASS. A failed earlier gate can still cause a later gate to be skipped. |
| `--debug` | During a real run, also copy full command output and the configuration values to the terminal. Completed logs receive this information without the option. |

`--gate`, `--tier`, and `--from` are alternative ways to choose gates; use
only one in a command. `--list` and `--check` cannot be combined with each
other. They also cannot be combined with a gate choice, a rerun option, or
`--dry-run`.

An unknown gate ID is a command error. The runner exits with code 2, suggests
nearby valid IDs, and runs nothing. Code 2 tells the terminal that the command
was not valid. The command also uses code 2 when a `--force-rerun` ID is not
part of the selected gates; it explains that the ID must be added to `--gate`
or removed from `--force-rerun`.

#### FAQ A2. What does the workstation check inspect?

The check confirms all of these conditions before a real board run:

1. The current Git history includes the board's required starting commit, one
   saved version of the repository.
2. There are no unsaved changes under `emulator/`, `ai/gates/`,
   `compute_data_vectors/`, `cobaya_theory/`, `syren/`, or the repository's
   top-level Python drivers.
3. Python can import PyTorch, CosmoLike, and Cobaya.
4. PyTorch can see a CUDA GPU.
5. The effective CoCoA root exists. It comes from the explicit `rootdir` value
   in `board_config.json`, or from `$ROOTDIR` when that value is `null`.
6. The configured driver folder and YAML folder exist.
7. The `debug` setting is exactly `true` or `false`.
8. `driver_fileroot`, the short name placed at the start of files made by a
   driver run, is present and is not placeholder text inside angle brackets.

`ai/gates/board_config.json` is allowed to differ because it stores
machine-specific paths. `ai/tests/` is not part of this particular unsaved-file
check; gates that use test files name them in their own file lists so a change
invalidates the related saved PASS.

This workstation check does not inspect every scientific input named under
`deploy_data` in `board_config.json`. After the workstation check, each
selected gate checks the input files it declares before starting its
calculation. A missing declared input records FAIL and may do so before an
attempt log exists; read the terminal output or the Detail cell in `BOARD.md`.

#### FAQ A3. Why did a gate refuse to start?

Read the first `[FAIL]` line printed by `--check`. Common examples are an
unset `$ROOTDIR` while `board_config.json` leaves `rootdir` as `null`, a path
in that file that does not exist, a missing CosmoLike import, a GPU that
PyTorch cannot see, or an unsaved file change in one of the checked folders.

The Git check does not require the current saved version to equal one exact
newest version. It requires the current history to include the board's fixed
starting commit, which is one saved repository version. Later saved versions,
including versions that add gate logs, are allowed.

### Appendices about results and files

#### FAQ B1. When may an earlier PASS be reused?

The runner reuses a PASS only when all of the following still match:

- the files executed by that gate;
- the input files and relevant configuration values;
- the saved log and its SHA-256 fingerprint, a calculated label that changes
  when even one byte in the log changes;
- the gate's required-result list; and
- every earlier gate on which it depends.

If one item differs, `--list` reports the relevant stale state and the next
ordinary run repeats that gate. This is why a word such as PASS alone is not
enough; the recorded evidence must still describe the current program and
input files.

#### FAQ B2. What happens after one gate fails?

One failure does not stop unrelated later gates. The board continues so the
workstation run can collect other independent results. A later gate that
needs an earlier gate without a current PASS is recorded as `SKIP-DEP` and
does not start. The earlier result may be FAIL, interrupted, out of date, or
not run.

The command exits with a nonzero code when any selected gate fails or is
skipped because a required gate lacks a current PASS. The Architect or another
program can therefore distinguish a fully accepted selection from a partial
run.

#### FAQ B3. Which files define and record the board?

| Path | Purpose |
| --- | --- |
| `ai/gates/board.py` | Registers each gate, its title, group, file list, earlier gates it needs, required results, and run function. |
| `ai/gates/run_board.py` | Parses commands, checks the workstation, selects gates, writes logs, and decides whether an earlier PASS is current. |
| `ai/gates/board_config.json` | Stores workstation paths, debug choice, small-run YAML paths, and optional older comparison commits. |
| `ai/gates/configs/` | Holds the small YAML files used by gate runs. |
| `ai/gates/checks/` | Holds focused programs that gates call for numerical or structural results. |
| `ai/gates/logs/` | Holds attempt logs, `BOARD.md`, and `board_status.json`. |

Gate run functions use the runner's logging helper when they launch a command.
The log records that command and everything it prints. A program called by the
gate may create temporary files, train models, or start another program. Those
internal actions appear in the gate log only when the called program prints
them.

#### FAQ B4. What is a golden comparison?

A golden comparison runs the same small configuration on the current code and
on a named older commit, then compares selected output lines exactly. A commit
is a saved version of the repository. The older commit is opened in a
temporary Git worktree, which is a separate folder for that saved version;
the runner removes that folder after the comparison.

Only the EMA identity gate has an older comparison commit configured by
default. Other gates use this extra comparison only when
`ai/gates/board_config.json` supplies one; otherwise they use their normal
small run.
