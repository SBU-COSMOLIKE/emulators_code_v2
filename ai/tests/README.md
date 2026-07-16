# Small tests for data handling, training rules, and AI tools

The Python files in this folder check one behavior at a time. For example, one
test confirms that a parameter table with one row keeps the two-dimensional
shape expected by the training programs. Another test damages a saved progress
file inside a temporary folder and confirms that the loader refuses it without
changing the original files.

The AI development roles normally run these tests while repairing or adding a
feature. A user can run the same commands to see the result. A test answers one
narrow question such as the two examples above. A gate decides whether the
complete set of evidence for a larger library promise is ready to accept. The
next section explains why both levels are useful.

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
table with one row keeps its two-dimensional shape. Small tests run often
while code is being written because a failure points close to the line that
needs attention.

A **gate** answers a larger question: is there enough evidence to accept one
promise about the library? A gate may run several tests and stand-alone
failure examples. It also checks that none of the required checks disappeared.
When the promise needs a GPU or configured scientific data, the gate records
that requirement. It reports `UNAVAILABLE` rather than calling an unperformed
check a pass.

The distinction is similar to work in a laboratory. One measurement checks
one part of an experiment. The final sign-off checks that every required
measurement was made under the right conditions. A test is the measurement;
a gate is the sign-off. A gate often calls tests from this folder, so the two
systems support each other.

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
that must be called directly. Each description follows the same order: the
file or small input created by the test, the action performed, and the result
that must be accepted or refused.

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
| `test_background_grid_contract.py` | Builds small background-data settings for `Hubble` in `km/s/Mpc` and `D_M` in `Mpc`, then sends those settings through the training geometry and the Cobaya reader. The correct pairs and finite offsets pass; an unknown unit, a Boolean or infinite offset, or altered saved center and scale values must stop before the program uses the background data. |
| `test_batching_sizing.py` | Creates a three-row toy batch with two inputs and either 7 or 14 stored target values. It checks the exact number of bytes required for `float32` and `float64` targets, confirms that this number changes the selected batch count, and requires a clear refusal when the available memory cannot hold one whole batch. |
| `test_cmb_checkpoint_axis.py` | Writes temporary CMB progress files containing the `tt`, `te`, `ee`, and `pp` spectra and the integer multipole values that label their columns. A correct axis loads every spectrum without changing the files; a missing, shifted, repeated, wrong-length, or wrong-number-format axis must stop loading before even the first spectrum is read. |
| `test_data_staging_paramnames.py` | Creates examples such as a numbered chain `train.1.txt` with shared names in `train.paramnames`, and a plain table with names beside its exact filename. It checks which names file wins, stops before opening a data vector when the names are absent or disagree, and confirms that all staging paths use the same maximum number of rows. |
| `test_d5_training_behavior_witnesses.py` | Runs small CPU calculations that stand in for the longer training gate. The checks pin the annealing values at every join, the learning-rate change after training only the final layers, the first moving-average record, and learning by the ReLU and Tanh models instead of a dead or mean-only control. They also calculate identifiers from the exact values in the earlier shared layers and confirm that those layers change during joint training but stay fixed while only the final layers learn. |
| `test_dataset_publication.py` | Builds a nested generated-data folder in a temporary directory, accepts it as a read-only saved generation, and then tries damaged files, extra files, links, path escapes, interrupted writes, and two writers switching the active generation. It requires every accepted file to match its recorded size and SHA-256 value, and it checks that later resume or append work begins from new private writable copies rather than modifying the accepted files. |
| `test_dataset_request_contract.py` | Builds uniform- and Gaussian-sampling requests for every supported scientific family, converts each request to the exact saved JSON bytes, and lists every file that request must generate. Changing a scientific field must change those bytes, while invalid bounds, duplicate parameter names, missing family files, or a wrong sampling field must refuse; settings used only to manage the write must not pretend to change the scientific request. |
| `test_finetune_post_step_and_provenance.py` | Trains a tiny one-weight model for one step and compares the order of three actions: the optimizer update, the pull back toward the saved starting weight, and the moving average of recent weights. The pull must occur after the update and before the average, and the fine-tune and transfer drivers must save the same facts about which earlier emulator supplied the starting weights. |
| `test_generator_checkpoint_refusal.py` | Creates complete, missing, and damaged progress-file sets for each data generator. Starting a new run may find no progress files, but an explicit resume or append request must require the exact files and axis lengths for that family; a bad requested set must stop instead of silently starting a new calculation, and chain-only mode must read only its parameter files. |
| `test_generator_member_binding.py` | Gives each real generator a family, variant, output filename prefix, and full or chain-only choice, then asks which absolute filenames it owns before any file is opened. The returned list must match that generator's actual progress-file methods; a family sent to the wrong generator, an invalid Boolean setting, or an unsafe filename prefix must stop before the program checks whether any candidate file exists. |
| `test_generator_payload_success.py` | Gives the generators small valid and invalid arrays, saves the valid arrays once, and reads them back from disk. A row becomes successful only when its shape, finite numbers, stored number format, and read-back values all match; overflow, a changed saved value, a wrong CMB spectrum count, wrong background keys, or matter-power files inconsistent with the selected mode must leave the row failed without rewriting an existing bad file. |
| `test_generator_run_control.py` | Tries the command-line switches for a new run, resume, append, full output, and chain-only output. Only native integer `0` or `1` values are accepted, append without resume is refused, the saved decision cannot later be edited, and chain-only output receives a different filename stem so it cannot overwrite a full generated set. |
| `test_grid2d_const_mask.py` | Creates a two-redshift by three-wavenumber Grid2D surface and marks the coordinates whose stored value must replace a network prediction. It checks that even an all-false mask is explicitly saved and restored, that a real low-wavenumber mask survives a save/load round trip, and that a missing, wrong-shaped, or wrong-number-format mask refuses instead of being guessed. |
| `test_grid2d_staging_row_contract.py` | Creates the same Grid2D rows once in memory and once in a disk-backed NumPy file, then selects rows with the same random seed. Both routes must preserve the identical selected order and must reject a claimed original row count that is zero, non-integer, or different from the rows actually present before allocating the staged result. |
| `test_parameter_table.py` | Writes one-row, multirow, reordered, and malformed parameter tables beside realistic `.paramnames` files. It checks that a one-row table stays two-dimensional, derived output columns are found by name, numbered chains use the correct shared names file, and duplicate names, missing columns, a wrong table width, an empty table, or any NaN or infinite value stops with a useful error. |
| `test_results_composition_mode.py` | Creates small in-memory emulator result files for plain output, NPCE output, frozen transfer, and refined transfer. Before a model is built, the reader must find the correct `plain`, `npce`, or `transfer` label, the required saved sections, and matching YAML settings after defaults have been filled in; missing labels, forbidden sections, wrong value types, or a label inferred only from which section happens to exist must refuse. |
| `test_results_const_mask_declaration.py` | Saves a Grid2D constant-coordinate mask and the SHA-256 value calculated from its exact ordered bytes. The result writer owns this fact: a caller may repeat the same declaration, but changing one mask position, changing the saved identifier, using it for non-Grid2D data, or omitting it from an older result format must refuse. |
| `test_results_rebuild_fixed_facts_names.py` | Creates an emulator result whose saved input names either agree or disagree across two internal records, then starts a rebuild. Matching names may reach model-weight loading; changing both records together must still stop first when the saved geometry has the original names, so compatible-looking model weights cannot hide a result file that describes different inputs. |
| `test_trf_token_width.py` | Builds tiny Transformer configurations with one feature or two features in each token. One feature is refused before ordinary layer construction because the old block produced an output independent of its input, while both supported two-feature configurations must still construct and respond to changed inputs. |
| `test_warmstart_perturbed_finite.py` | Feeds valid, NaN, and infinite comparison rows to the fine-tune and transfer warm-start paths. Valid rows keep the prior numerical result; a bad input after conversion or a bad output after conversion back to physical values must report the correct row and quantity immediately. Deliberately removing either check must make the test fail by reproducing the older misleading diagnosis. |

## AI workflow and policy test inventory

These modules check the rules used by the Architect, Implementer, Red Team,
mailbox watcher, and repository protection tools.

| File | What it checks |
| --- | --- |
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
| `tools_mailbox_daemon_output_style_repro.py` | Runs ordinary waits, refusals, and progress reports and compares the exact terminal text with the examples in the README. The check catches unclear separator lines, unexplained all-capital messages, wrong waiting counts, or documentation that promises output the program no longer prints. |
| `tools_mailbox_daemon_primary_worktree_repro.py` | Uses temporary Git repositories to create and then reuse one separate Claude work folder and one separate Sol work folder. It checks the saved folder records, simultaneous first-time setup, movement of older mailbox files into the selected folders, and refusal when an existing branch or folder name would point at the wrong place. |
| `tools_mailbox_daemon_redteam_repro.py` | Sends several Red Team messages while preview mode and competing watcher processes are active. A message must be completely written before it becomes visible, only one watcher may claim it, message numbers must remain unique, and malformed bodies or unreplaced placeholder text must refuse without losing another message. |
| `tools_mailbox_daemon_rendezvous_repro.py` | Starts short-lived fake roles and observes the watcher's safe-stop countdown after all started roles finish; this finished point is what the command calls a cycle boundary. It checks `--cycle 0`, positive cycle counts, new messages arriving during the countdown, two simultaneous senders, and refusal when the backlog file cannot be understood. |
| `tools_mailbox_daemon_role_models_repro.py` | Selects model names and aliases for the Architect and Implementer, then inspects the exact commands the watcher would launch for each role. Defaults, valid aliases, invalid names, and role routing must agree, and changing a copied daemon source file must stop the run rather than launching an unverified command. |
| `tools_mailbox_daemon_staleness_repro.py` | Starts fake role commands that finish, time out, retry, or leave a request in the `inflight/` folder, which holds a message while one role is working on it. It checks finished-file order and recovery, and it requires refusal when a program file changes during the run, a required file becomes a link, a timeout record is malformed, or a copied daemon no longer matches its approved bytes. |

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
