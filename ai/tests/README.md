# Developer tests and reproductions

This folder checks that a code change keeps a previously accepted behavior.
Most files run on the CPU and finish without starting a training job. The
longer files rebuild tool failures inside temporary folders so they do not use
the live AI mailbox, the project folders that hold requests between roles.

For tests that train a model or require configured scientific data, use the
[`ai/gates/` board](../gates/README.md) instead.

## Contents

1. [Run the usual test set](#run-the-usual-test-set)
2. [Choose a test, reproduction, or gate](#choose-a-test-reproduction-or-gate)
3. [Read the result](#read-the-result)
4. [What do these files use or change?](#what-do-these-files-use-or-change)
5. [Find the file that covers a behavior](#find-the-file-that-covers-a-behavior)
6. [Scientific and data test inventory](#scientific-and-data-test-inventory)
7. [AI workflow and policy test inventory](#ai-workflow-and-policy-test-inventory)
8. [Direct reproduction inventory](#direct-reproduction-inventory)
9. [How agents use these files](#how-agents-use-these-files)
10. [Add or update a test](#add-or-update-a-test)

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
is a named acceptance check. The validation board lists and runs those gates.

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

Use the tables below when the search term appears in several files. The first
two tables list modules found by `unittest discover`. The third lists scripts
that must be called directly.

## Scientific and data test inventory

These modules check numerical rules, training setup, generated data, and saved
emulator files. Grid2D means data arranged on a redshift-by-wavenumber grid.
A warm start begins from a saved emulator rather than new random weights.
SHA-256 is a fixed-length identifier calculated from exact file bytes.

Neural polynomial chaos expansion (NPCE) combines a frozen polynomial base
with a neural model that learns the remaining difference. A checkpoint is a
saved generator-progress file. A saved group is one named section inside an
emulator result file.

| File | What it checks |
| --- | --- |
| `test_background_grid_contract.py` | Accepted background quantity/unit pairs, finite offsets, saved geometry facts, and agreement with the Cobaya background reader. |
| `test_batching_sizing.py` | The target array's exact byte cost in batch sizing and refusal when one complete batch cannot fit. |
| `test_cmb_checkpoint_axis.py` | The exact CMB multipole axis saved with a progress file and refusal of missing, shifted, repeated, or malformed axes before spectra are read. |
| `test_data_staging_paramnames.py` | How `.paramnames` files are chosen for numbered chains and plain data tables, plus the shared maximum row count for loading data. |
| `test_dataset_publication.py` | Generated-data folders that cannot change after acceptance, SHA-256 checks of each file, competing writers, crash points, path escape attempts, links, and corrupted files. |
| `test_dataset_request_contract.py` | The exact scientific request fields and generated member names for each supported family and sampling mode. |
| `test_finetune_post_step_and_provenance.py` | The pull toward saved fine-tune weights happens after the optimizer step, the moving weight average sees that pull, and both drivers save the same origin facts. |
| `test_generator_checkpoint_refusal.py` | Missing or corrupt requested progress files refuse instead of silently starting fresh, and each family names its required files and axes. |
| `test_generator_member_binding.py` | Each generator binds its real driver, output family, variant, and exact progress-file names before it looks for an existing file. |
| `test_generator_payload_success.py` | A generated row is marked successful only after its array shape, finite values, stored number format, and exact contents are checked again after saving; each family also checks its expected array shape. |
| `test_generator_run_control.py` | The three generation operations, chain-only choice, strict integer switches, and the complete saved run-control record. |
| `test_grid2d_const_mask.py` | Construction, storage, and validation of the Grid2D constant-column mask. |
| `test_grid2d_staging_row_contract.py` | Exact original row counts and preservation of one randomly selected row order for memory-resident and disk-backed Grid2D data. |
| `test_parameter_table.py` | Strict `.paramnames` parsing, input/output column selection, derived columns, widths, and finite numeric values. |
| `test_results_composition_mode.py` | Agreement among the named sections saved in the result, the YAML settings after defaults are filled in, and the exact `plain`, `npce`, or `transfer` label before model construction. |
| `test_results_const_mask_declaration.py` | A caller cannot replace the writer's Grid2D constant-column fact, and its exact saved SHA-256 value is checked. |
| `test_results_rebuild_fixed_facts_names.py` | Saved input names are checked before model weights are loaded during rebuild. |
| `test_trf_token_width.py` | Transformer models refuse one feature per token and retain valid behavior at two features per token. |
| `test_warmstart_perturbed_finite.py` | Warm-start comparison rows name NaN or infinite encoded inputs and outputs instead of misreporting another quantity. |

## AI workflow and policy test inventory

These modules check the rules used by the Architect, Implementer, Red Team,
mailbox watcher, and repository protection tools.

| File | What it checks |
| --- | --- |
| `test_handoff_contract.py` | Required sections and evidence in Architect and Red Team instructions, UTF-8 source-note limits, severity, character limits, and the single program allowed to validate those instructions. |
| `test_mailbox_conditional_preamble.py` | Terminal prompts require a reply only when another mailbox message is expected. |
| `test_mailbox_daemon_architect_entrypoint.py` | The public send and ping commands address only the Architect while the watcher can still send approved internal messages to other roles. |
| `test_mailbox_daemon_severity.py` | The high, medium, or low threshold for new problems; default medium behavior; the same setting reaching each launched role; and fix-only or disabled-Red-Team modes overriding it. |
| `test_permanent_note_guard.py` | The SHA-256 identifier for exact permanent-note bytes detects changes in current files, files selected for the next commit, saved commits, extra note files, files replaced by links to other paths, or the guard program. |
| `test_permanent_note_style_contract.py` | Permanent notes keep neutral current-language rules, unique Markdown link targets, and no dated ticket diary. |
| `test_role_directive_contract.py` | The user contacts only the Architect, thinking roles supply detailed instructions, and execution roles cannot edit permanent notes. |
| `test_tests_readme_inventory.py` | The inventory tables name every immediate Python file in `ai/tests/` and name no file that is absent. |
| `test_ticket_change_guard.py` | Exact added-plus-deleted character counting, a valid saved starting version, a clean saved result when a positive limit is active, limits on how much file content the counter reads, and refusal of binary or non-UTF-8 changes. |

## Direct reproduction inventory

These scripts are not found by `unittest discover`. Each script has a
`main()` starting function, prints its own checks, and returns a nonzero exit
code when its witness fails.

| File | What it rebuilds inside temporary folders |
| --- | --- |
| `finite_contract_cuda_wording_repro.py` | A read-only check that the finite-contract gate's failure message names the machine still required to compile and run CUDA. |
| `tools_backlog_bundle_repro.py` | Backlog packing and import through `.tar.xz`, unchanged file contents after packing and unpacking, existing outputs, changed protected files, links, archive names that escape the import folder, malformed archives, and size limits. |
| `tools_handoff_router_repro.py` | Manual handoff routing in temporary repositories: working folder, sequence, lock, clipboard, directive, evidence, character-limit, severity, role, and source-note refusals. |
| `tools_mailbox_daemon_dead_mailbox_repro.py` | Warnings when no watcher is running, held or outdated locks, read-only preview, other running watchers, links that point a required folder elsewhere, pipe files that can block a reader, and send/ping behavior. |
| `tools_mailbox_daemon_fix_only_repro.py` | Whether fix-only mode starts repair requests while leaving new-problem requests waiting, what happens when two terminals try to enable that mode at the same time, the same rule reaching each started role, runs with only an Architect and Implementer, deliberately damaged source copies, and README/help agreement. |
| `tools_mailbox_daemon_landing_debt_repro.py` | Detection of an accepted fix that has not reached `main`, creation of only one repair request, locks, malformed saved state, recovery, and repeated watcher passes. |
| `tools_mailbox_daemon_max_repro.py` | The ticket `--max` character limit, default zero, accepted and refused values, inherited settings, terminal report, and the same limit reaching every started role. |
| `tools_mailbox_daemon_no_redteam_repro.py` | Two-role watches, deferred Sol requests, work-folder separation, combined fix-only mode, lock cleanup, simultaneous sends, and deliberately damaged source copies. |
| `tools_mailbox_daemon_output_style_repro.py` | Exact terminal refusal, waiting-count, and progress text; matching README examples; and detection of separator or all-capital wording changes. |
| `tools_mailbox_daemon_primary_worktree_repro.py` | Creation and reuse of separate Claude and Sol work folders in temporary Git folders, checks of saved folder records, simultaneous setup, movement of old mailbox files into the new work folders, and name-collision refusal. |
| `tools_mailbox_daemon_redteam_repro.py` | Read-only preview, only one watcher taking a message, complete file writes before a message becomes visible, unique message numbers across roles, malformed message bodies, and literal placeholder text. |
| `tools_mailbox_daemon_rendezvous_repro.py` | The watcher's safe-stop countdown, the point after started roles finish, `--cycle` values, waiting work, simultaneous sends, and malformed backlog refusal. |
| `tools_mailbox_daemon_role_models_repro.py` | Architect and Implementer model defaults, aliases, validation, command construction, role routing, and deliberately damaged source copies. |
| `tools_mailbox_daemon_staleness_repro.py` | Program-file changes during a run, timeouts, retries, finished-file ordering, requests left in the work-in-progress `inflight/` folder, required files replaced by links to other paths, malformed timeout records, and deliberately damaged source copies. |

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
