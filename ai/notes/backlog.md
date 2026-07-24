# Execution backlog

This operational record is tracked in Git so unfinished fixes survive a new
clone. Only the Architect updates it. The daemon includes the Architect-sealed
ticket update in the same landing commit as the accepted fix.

Unfinished work appears first and completed work appears second. Each ticket
begins with an explanation for a human reader. Exact commits, tests, branches,
and internal identifiers appear later in the ticket under **Technical record
for development tools**.

## Contents

- [Open tickets](#open-tickets)
- [Parked edge cases](#parked-edge-cases)
- [Closed tickets](#closed-tickets)

## How to read this backlog

The Architect maintains this file. The Implementer and Red Team may read it,
but they never edit it. The daemon saves a sealed ticket update with the
accepted fix instead of creating a separate policy landing.

The watcher counts lines that begin exactly with `- OPEN`. Each index line also
records `BUG FIX` or `NEW FUNCTIONALITY`; this lets the watcher distinguish an
existing defect from a feature with the same priority. The linked list below
contains one such line for each unfinished ticket. Do not add another
`- OPEN` line inside a ticket.

Every ticket also shows a **Red Team reopen count** and whether another Red
Team reopening is allowed. The count starts at zero and never resets. One
cycle always belongs to one ticket. In the normal three-role run, the cycle
includes the Architect and Implementer exchange, the accepted commit, and one
Red Team review of that exact commit. The Architect may start another ticket
while the review runs only when the current watch has another unused cycle;
`--cycle 1` never authorizes a second ticket.

If that review reports `REOPEN`, the Architect restores the ticket to Open,
adds one to the count, and preserves the Red Team's detailed finding without
reproducing the bug immediately. Later, when the ticket reaches the front of
the permitted order, the Architect gives the evidence GO or NO-GO. GO keeps
the ticket open for repair. NO-GO closes the ticket and permanently changes
its reopening line to **barred by Architect NO-GO**. The Red Team may not
reopen that ticket again; a genuinely different defect requires a new ticket.
After the second reopening, the Architect demands increasingly specific new
evidence. The sixth reopening automatically makes the ticket Low so a repeated
disagreement cannot consume the whole work queue.

Red Team review is advisory. The Architect may accept and commit an
Implementer fix without Red Team approval. A Red Team `REOPEN` creates later
work but does not undo the commit. A finite watch still refuses to start any
ticket beyond the number selected with `--cycle`.

Only the Architect writes this file. A saved SHA-256 fingerprint detects an
unexpected edit before the Architect writes again. The Implementer and Red
Team may read the backlog but never edit it or replace its fingerprint.

## Words used in open tickets

A **saved emulator** is the pair containing learned model weights and the
scientific record needed to interpret those weights. The technical record may
call this pair an **artifact**. A **saved-file format** or **schema** states
which fields that pair must contain.

An **identity** is a saved fingerprint of the exact inputs, settings, formulas,
or files that produced an object. To **publish** a file means to validate a
complete temporary file and then place it at the final name a reader uses. An
**authenticated** group of files has recorded digests that prove the files
belong to the same completed result.

An **adapter** is the Python bridge that gives Cobaya a result from a trained
emulator. A **checkpoint** is one saved state from the middle of training. A
fine-tune **anchor** limits how far new weights may move from their starting
weights. **Provenance** is the saved record of where data or weights came from.

A model's **domain** is the range of physical inputs on which it may be used.
**Composition** is the formula that combines an emulator correction with an
analytic base calculation. A **resolved run record** stores the settings that
the program actually used after defaults and automatic choices were applied.

The recurring scientific abbreviations are CMB for cosmic microwave
background, MPS for matter power spectrum, and PCE for polynomial chaos
expansion, a polynomial emulator. CUDA is NVIDIA's accelerator platform. A
**gate** is a named final check for a larger requirement, while a **test** asks
one narrow question.

CPU means the computer's general processor, while GPU means an accelerator.
MPI is the message-passing system used to coordinate several generator
processes. CAMB is the upstream cosmology program that provides reference
calculations. HDF5 is the structured `.h5` file format, and YAML is the text
format used for user settings. EMA means weights averaged across recent
training steps, and CNN means convolutional neural network. In a CMB ticket,
TT and EE are auto-spectra and TE is their cross-spectrum.

CosmoLike is the upstream program that evaluates several survey observables,
and Syren is an analytic matter-power calculation used by this repository.
`H0` is the present-day Hubble constant and `h = H0 / 100` is its standard
dimensionless form. ReLU and Tanh are activation curves inside a neural
network, while BerHu is an error measure used during training. A moving
average smooths model weights across recent training steps. A transformer is a
model that mixes information with attention; FiLM is a learned scale-and-shift
operation applied inside a model.

# Open tickets

The Architect assigns priority when a ticket is admitted and records the
reason. Critical is an Architect-only bug classification for evidence that a
current defect broadly breaks a central library workflow or systematically
makes the library's scientific results invalid. High means a bug can make the
science wrong, lose data, halt a core operation, or severely damage core
behavior. For every High ticket, the Architect must state the demonstrated
impact and why Medium is not sufficient. Urgency, a missing test, unfinished
cleanup, or an expensive check does not by itself make a ticket High. The Red
Team uses the same restraint when proposing a priority. Medium means a
concrete problem is reasonably likely during normal work but does not meet the
High boundary. Low includes concrete but improbable edge cases.

“The science can be wrong” is not sufficient by itself for High. The defect
must threaten a central scientific calculation, the training data, a served
emulator result, or another primary library result. A defect limited to a
plot, diagnostic ranking, optional report, or other supporting analysis is
normally Medium even when its output is misleading. Promote such a defect
only when evidence shows that it also corrupts a primary result or blocks a
core workflow.

Every ticket is also a Bug fix or New functionality. Severity is the first
sorting decision; ticket type is the second. The complete order is Critical
bugs, High features, High bugs, Medium bugs, Medium features, Low bugs, and
Low features. A Low bug therefore never jumps ahead of a Medium feature, while
a Medium bug comes before a Medium feature. Features may be High, Medium, or
Low but never Critical. The words “after the backlog is closed” create a Low
feature whose prerequisites are all tickets that were already open when it was
admitted.

A blocked ticket stays in its group with the blocker; the Architect may move
to the next permitted ticket while required hardware, data, an external
decision, or a named prerequisite is unavailable. New evidence may change a
bug's severity, but the Architect records the reason for every upgrade or
downgrade. No ticket is promoted to Critical to change the active roles.

A bounded repair may close an actionable bug when it removes the ticket's
demonstrated failure and leaves only a harmless exceptional case below Low. If
complete
coverage would add disproportionate complexity, the simpler result is
acceptable. The Architect records the exact remainder under **Parked edge
cases** without claiming complete coverage.

A parked **LOW — EDGE CASE** has no `- OPEN` line, never enters a watcher count,
and is not a `--severity` choice. Only a user request that explicitly names the
ticket and asks the Architect to solve it may activate it as ordinary Low work.
This class never hides a probable failure, wrong primary science, data loss,
or broken core operation.

Backlog counts never change a role. Sol is the advisory Red Team when enabled
and does not implement tickets. Parallel work comes from the normal pipeline:
the Implementer may code a newly admitted ticket while the Architect audits a
previous commit and the Red Team reviews an earlier accepted commit. This
overlap is allowed only when the finite watch has another unused ticket slot.
Each ticket still consumes exactly one cycle.

## Open ticket index

### Critical

No open CRITICAL tickets.

### High

High new functionality appears before High bug fixes. No High feature is
currently open.

No open HIGH tickets.

### Medium

Medium work begins only after the permitted High work above.

- OPEN **MEDIUM** **BUG FIX** — [Write the GetDist posterior column with the correct meaning](#open-getdist-column)
- OPEN **MEDIUM** **BUG FIX** — [Publish structured study and diagnostic results](#open-study-diagnostics)
- OPEN **MEDIUM** **BUG FIX** — [Run real hardware checks for training behavior](#open-training-hardware)
- OPEN **MEDIUM** **BUG FIX** — [Run saved PyTorch compilation settings on CUDA](#open-compile-modes)
- OPEN **MEDIUM** **BUG FIX** — [Complete older cross-family workstation checks](#open-workstation-debt)
- OPEN **MEDIUM** **BUG FIX** — [Finish real workstation checks for the current saved-file format](#open-schema-v3-gate-fixtures)
- OPEN **MEDIUM** **BUG FIX** — [Save every effective setting and reset each repeated study](#open-resolved-run-record)
- OPEN **MEDIUM** **NEW FUNCTIONALITY** — [Finish safe fine-tuning against the original weights](#open-finetune-anchor)
- OPEN **MEDIUM** **NEW FUNCTIONALITY** — [Retry failed generator rows reproducibly](#open-generator-failure-retry)
- OPEN **MEDIUM** **NEW FUNCTIONALITY** — [Record which physics formulas produced each dataset and trained emulator](#open-physics-implementation-identity)
- OPEN **MEDIUM** **NEW FUNCTIONALITY** — [Refuse polynomial-emulator requests outside the fitted parameter range](#open-pce-domain-enforcement)
- OPEN **MEDIUM** **NEW FUNCTIONALITY** — [Add advertised CMB unit and multipole conversions](#open-cmb-serving-conversions)

### Low

- OPEN **LOW** **BUG FIX** — [Make tracked explanations describe one coherent current library](#open-python-prose-review)

<a id="open-mps-test-import-isolation"></a>
## Isolate the matter-power adapter test without replacing imported modules

### High-level summary

The matter-power sigma-eight test replaces three entries in Python's shared
import table while it loads the Cobaya adapter. The context manager restores
the table, but imported `emulator` submodules remain attached to their parent
package. A later test can then hold one module object while production imports
another object with the same name.

Running the sigma-eight test before the dark-energy generator test fails two
spy assertions. The reverse order and the generator test alone pass. The test
suite is therefore order dependent and can report a false failure after an
apparently temporary replacement has ended.

### Current status

**Ticket type: BUG FIX.**

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**CLOSED.** The in-process loader that replaced `sys.modules` entries is
removed. The numeric sigma-eight checks and the dark-energy adapter checks —
the second consumer of the same loader — now run in child processes launched
by their discovery-visible test files. Each child imports the adapter through
the on-disk stand-in package `ai/tests/cobaya_minimal_stub/`, placed first on
the child's PYTHONPATH before the child starts, so the parent process's import
table and `emulator` package attributes are never edited. The launcher also
runs one negative control with a deliberately wrong known answer; that child
run must fail. Both explicit module orders with the dark-energy generator test
pass, and the moved dark-energy check now proves wa sensitivity against the
real Syren base instead of a replaced function.

**Severity: HIGH.** The leak makes independent validation depend on test order
and can produce a false gate result. It does not alter normal emulator runtime
or scientific output, so it does not meet the Critical boundary.

### What is already fixed

Everything: the loader, both consumers, and the negative control described
below.

### What is missing

Nothing for this ticket.

<details><summary>Technical record for development tools</summary>

Former owner: `ai/tests/test_mps_sigma8_contract.py::_load_mps_adapter`,
also imported by `ai/tests/test_mps_dark_energy_adapter.py`. Replacement:
`run_isolated_child_checks` launches `ai/tests/mps_sigma8_child_checks.py`
and `ai/tests/mps_dark_energy_child_checks.py` with
`PYTHONPATH=cobaya_minimal_stub:ROOT`; each child refuses to run when the
imported cobaya lacks the `COBAYA_MINIMAL_STUB` marker. Verified: both
explicit orders with `ai.tests.test_generator_dark_energy_facts` pass, the
full discovery suite passes, the negative control
(`MPS_SIGMA8_EXPECTED_OVERRIDE=0.9`) fails the child, and each launcher
asserts `Ran 5 tests` plus an unchanged parent `sys.modules`.

</details>

<a id="open-artifact-drift-import-isolation"></a>
## Test saved activation defaults without replacing a live function

### High-level summary

The saved-artifact drift test changes the default of the live
`make_activation` function while Python is running. The test is meant to prove
that rebuilding reads the saved gate count instead of a current source-code
default, but changing a shared function can leak into an unrelated test when
cleanup or target selection is wrong.

The same scientific check can run in a separate process with a visible
test-only source default. That preserves the useful negative control without
changing executable behavior shared by the rest of the test process.

### Current status

**Ticket type: BUG FIX.**

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**CLOSED.** The drift proof no longer edits `make_activation.__defaults__`
in a running process. The gate copies the emulator package into its
temporary folder, changes only the `n_gates` default line on disk, and
rebuilds the plain save in a child process whose PYTHONPATH names the copy
first. The child's first step verifies the changed default is live and
refuses with a dedicated exit code otherwise, so a launch that imported the
ordinary package cannot pass as a proof. The bitwise prediction comparison
is unchanged, and a focused CPU test exercises the same helpers on a small
synthetic gated-power artifact, including the unmodified-copy refusal.

**Severity: HIGH.** The patch changes a function used by artifact rebuilding
and can weaken or contaminate acceptance evidence. It does not change normal
runtime unless the test leaks, so it does not meet the Critical boundary.

### What is already fixed

Everything: the child-process observation, the live-default control, and the
local CPU verification described below.

### What is missing

Nothing for this ticket. The full gate run on the configured workstation
remains owed under
[Complete older cross-family workstation checks](#open-workstation-debt).

<details><summary>Technical record for development tools</summary>
Former owner: the `__defaults__` save/replace/restore in
`ai/gates/checks/gsv_bitwise_drift.py::main`. Replacement helpers in the
same file: `prepare_drift_source_copy` (one-line substitution, refuses
unless the anchor line appears exactly once), `run_drift_child`
(PYTHONPATH-selected copy, probe and output files), and `drift_child_main`
(exit 3 without the live modified default). Local evidence:
`ai/tests/test_drift_gate_child_isolation.py` — copy differs in exactly one
file, child rebuild of a gated-power save is bitwise-equal, an unmodified
copy is refused, and the synthetic fixture rebuilds in-process. The durable
behavior is described by `save-rebuild-drift.code-default-drift-ignored` in
`ai/notes/artifacts-inference-warmstart.md`, updated with this repair.
</details>

<a id="open-finite-cycle-admission"></a>
## Make a finite watch start exactly the requested number of tickets

### High-level summary

The `--cycle` option is the human's limit on how many tickets one watcher may
work on. With `--cycle 1`, the current program can start a second ticket while
the first ticket waits for its Red Team review. It can then count the first
cycle complete and exit with the unrequested second ticket already changed or
partly completed.

This defeats the main cost and runtime limit of the mailbox system. It can
spend model credits and modify a worktree beyond the number of tickets the
human authorized.

### Current status

**Ticket type: BUG FIX.**

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**CLOSED.** Commit `20119a1` reserves finite capacity before a public Architect
turn, converts only an exact Implementer ticket, and releases a valid
non-ticket control outcome without counting a cycle. A later request remains
byte-for-byte untouched when the requested limit is full. Architect and Red
Team turns now refuse tracked or untracked source edits. The parent daemon
owns candidate landing, restart recovery, push debt, and clean role-baseline
synchronization.

**Severity: CRITICAL.** A finite watch can start work beyond the human's
explicit bound and then exit with that extra ticket still active. High is
insufficient because this broadly breaks the central control that limits
unattended edits, runtime, and model-credit use.

### What is already fixed

The watcher waits for running role processes before an ordinary safe stop.
Completed cycle returns are saved so a restart does not lose an already
finished cycle.

### What is missing

Nothing for this ticket. Safe continuation after `main` legitimately advances
beyond an already prepared landing remains the separate Medium ticket
[Recover safely when main advances after a landing is prepared](#open-stale-landing-reaudit).

<details><summary>Technical record for development tools</summary>

Witness: `SafeKillRendezvous.begin_attempt()` checks only completed returns;
`register_ticket_cycle_message()` does not reserve positive-cycle capacity;
and the positive exit predicate does not reject another active ticket. The
repair needs durable or restart-reconstructed reservations, pre-claim
deferral, a one-ticket completion rule for both role setups, independent
Architect and Implementer worktrees, and focused tests for normal, two-role,
restart, pipelined, and over-limit queues. Automatic severity thresholds and
role-changing emergency modes are removed rather than translated.

The late-admission witness uses two root `to-fable` user requests, so an
Opus-only reservation is not sufficient. The integrity witnesses create
`emulator/architect_created.py` and `emulator/redteam_created.py` inside their
respective saved role worktrees and require both turns to be refused without
confusing ignored transport output with source.

</details>

<a id="open-architect-note-landing"></a>
## Land Architect-owned permanent-note commits before later tickets use them

### High-level summary

Only the Architect may change the eleven permanent notes. The Architect can
commit an accepted policy update in the coordination worktree, but the current
watcher has no safe operation that moves that note-only commit onto `main`.

Leaving the commit on the coordination branch does not merely delay the
documentation. The Implementer worktree remains at the old `main`, so the
next ticket refuses to start from the Architect's newer commit. If the next
ticket starts from the old commit instead, the permanent policy change never
reaches the candidate or GitHub.

The reverse mismatch is also unsafe. After a normal source-code ticket lands,
the user's `main` may contain the new daemon and role rules while the saved
Architect worktree still contains the old versions. The next command then
re-executes old coordination code and can undo the reliability gained by the
accepted ticket.

### Current status

**Ticket type: BUG FIX.**

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**CLOSED.** Commit `20119a1` adds the narrow Architect-only B/P landing route,
restart journal, bounded push-debt record, and clean role-baseline update. The
route is cycle-free, cannot be used by Implementer or Red Team, and refuses to
mix a permanent-note transition with an ordinary ticket.

**Severity: CRITICAL.** The permanent notes control every later Architect,
Implementer, and Red Team instruction. A valid Architect policy update can
currently halt later tickets or disappear from the shared history, breaking
the central authority mechanism rather than one optional diagnostic.

### What is already fixed

The permanent-note guard proves that the Implementer and Red Team did not
change the protected files. The Architect can review their complete diff and
create a separate note-only commit in the coordination worktree.

### What is missing

Nothing for this ticket.

<details><summary>Technical record for development tools</summary>

Owners: `ai/tools/mailbox_daemon.py`, the role directives, and the real-Git
primary-worktree reproduction. Required message: a new exact
`architect-notes-go` request containing full B and P hashes. Required negative
cases include a non-note path, multiple parents, changed or dirty user main,
dirty or moved coordination `HEAD`, an active ticket, malformed/replayed
requests, and attempts by another role to use the route.

</details>

<a id="open-dataset-continuation-features"></a>
## Continue generated datasets exactly and manage old generations

### High-level summary

After the publication bug is repaired, a user should be able to stop an
expensive generator and later obtain the same additional rows that one
uninterrupted run would have produced. Training should also be able to select
one named completed generation while older generations are retained or
removed by an explicit rule.

The building blocks can make a private continuation copy, but the production
generator does not yet save every random state, walker position, or selection
decision needed for exact continuation. It also lacks the complete user policy
for pinning consumers and removing old generations.

### Current status

**Ticket type: NEW FUNCTIONALITY.**

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**CLOSED.** The publication framework this ticket extended was removed when
`compute_data_vectors/` returned to plain files: a generator run's outputs
are ordinary files under `chains/`, there are no generations to pin or
retire, and append now draws from a stream derived from the seed plus the
existing row count, so it is reproducible and never repeats saved rows. The
remaining idea this ticket described — an append whose rows are bitwise
identical to one uninterrupted longer run — is retired with that framework:
it required persisting complete sampler state and is not needed for a
reproducible dataset.

<details><summary>Technical record for development tools</summary>
Owners: `compute_data_vectors/generator_core.py`, MPI coordination, generation
manifests, and training staging. The High publication repair is closed; this
feature must not weaken its fail-closed rules.
</details>

<a id="open-finetune-anchor"></a>
## Finish safe fine-tuning against the original weights — Unit 24

### High-level summary

Fine-tuning improves an existing emulator while an anchor limits how far each
new weight may move from its starting value.

The CPU checks pass, but a real GPU run has not proved that the anchor covers
every trainable weight or that the saved result rebuilds correctly.

Closing this ticket early could publish a model that moved outside its intended
constraint or whose recorded source cannot be verified after loading.

### Current status

**Ticket type: NEW FUNCTIONALITY.**

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**OPEN.** The code candidate and CPU checks are complete, but the required
CosmoLike GPU evidence is unavailable on this computer.

**Priority: MEDIUM.** The anchor is deliberately unavailable in production,
so no accepted primary emulator is currently shown to violate it. Enabling
the feature safely and obtaining its GPU evidence are important, but they do
not justify emergency High-bug work.

### What is already fixed

The on-main slice uses one provenance assembler for scalar and shared-family
fine-tuning. A separate local candidate implements eager and compiled
parameter coverage, mask and frozen-name checks, the executed artifact
record, and the final readback gate.

### What is missing

Run the exact candidate on a supported GPU with real CosmoLike
`finetune-smoke`, then rebuild and read the saved artifact on the CPU. After
those results pass, the Architect must update the relevant permanent note and
merge the candidate.

<details>
<summary>Technical record for development tools</summary>

- Partial on `main`: `2742156`; focused tests 4/4 and AI tests 247/247.
- Complete local candidate: `25ac6d9` on
  `codex/unit24-anchor-hardware`; isolated CPU tests 299/299.
- Candidate evidence also includes the expanded identity gate, board
  self-test, compilation, whitespace check, and independent implementation
  review.
- Release remains `NO-GO` until the real GPU smoke and artifact readback pass.

</details>

<a id="open-training-hardware"></a>
## Run real hardware checks for training behavior — DIDACTICS-62

### High-level summary

Five checks measure learning-rate changes, loss and moving-average schedules,
two activation functions, and which model layers update in the second phase.

CPU runs prove the checks' arithmetic, but the complete CUDA drivers have not
run with real CosmoLike data on the configured workstation.

Until that run passes, a change may look correct locally while production
training updates the wrong layers or follows the wrong schedule.

### Current status

**Ticket type: BUG FIX.**

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**OPEN.** The implementation is on `main`; only the configured workstation
run is missing.

**Severity: MEDIUM.** No defect is confirmed by the completed CPU checks, but
ordinary GPU training remains unverified and may expose a device-specific
schedule or layer-update error.

### What is already fixed

The five gates have independent numerical judges and local refusal controls.
The CPU children, board planning, registry, self-test, compilation, and note
guard all pass.

### What is missing

Run the real five-gate selection on a CosmoLike/CUDA workstation. The result
must include the learning-rate cadence, first moving-average record,
joint-versus-frozen trunk digests, and both full ReLU and Tanh driver runs.

<details>
<summary>Technical record for development tools</summary>

- Landed and pushed as `03723c8`.
- Local evidence: 28/28 focused tests, 319/319 full AI tests, all CPU child
  required results, five-gate dry run, board list/self-test, compilation,
  whitespace check, exact-commit permanent-note guard, and independent GO.
- A Mac CPU result must not be recorded as the missing workstation evidence.

</details>

<a id="open-compile-modes"></a>
## Run saved PyTorch compilation settings on CUDA — Unit 93

### High-level summary

A saved emulator records the PyTorch compilation setting needed when its model
is rebuilt.

Local checks show that two saved settings reach the compiler, but neither
rebuilt result has completed the required CUDA execution check.

The files may therefore pass CPU inspection and still fail when a user loads
the emulator on a GPU.

### Current status

**Ticket type: BUG FIX.**

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**OPEN.** The implementation is landed; the two real CUDA rebuilds are owed.

**Severity: MEDIUM.** No CUDA defect has been demonstrated: the implementation
and CPU controls pass, but the required real-device validation is still
missing. That evidence is not enough for High because a missing hardware run
is validation debt, not proof that a core workflow is broken.

### What is already fixed

The schema-v3 reader consumes `compile_mode` and refuses its absence. Local
controls cover lost, duplicate, swapped, hard-coded, raising, identity, and
discarded compiler results.

### What is missing

On a CUDA workstation, rebuild both persisted modes with
`compile_model=True`. The child must exit 0 and both required results must
PASS; the preferred extra check also changes each real hard-coded source path
and confirms that the related test fails.

<details>
<summary>Technical record for development tools</summary>

- Landed and pushed as `18560d3`.
- Local evidence: 58/58 AI tests, 12/12 schema and verdict controls, CPU PASS,
  CUDA UNAVAILABLE with honest return code 2, board self-test, identity-family
  regressions, compilation, diff check, and two independent reviews.
- Durable audit owner: `ai/notes/gates-and-board.md`, Unit-93 current-schema
  implementation audit.

</details>

<a id="open-workstation-debt"></a>
## Complete older cross-family workstation checks

### High-level summary

Several completed repairs still need real-workstation checks, including
rebuilding a saved emulator, reading a real CMB calculation, and running an
optimization study with its persistent journal.

Their CPU and source checks passed, but the required CoCoA workstation is not
available on this computer.

An input, device, or saved-file mismatch may therefore remain hidden until a
user starts a real training or inference job.

### Current status

**Ticket type: BUG FIX.**

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**OPEN.** The required software and data are available only on the configured
workstation.

**Severity: MEDIUM.** The local checks found no confirmed defect, but normal
saved-emulator, CMB, covariance, and study operations still lack their final
configured-workstation evidence.

### What is already fixed

The underlying artifact, optimization-study, saved-state, covariance, and AI
tree changes are on `main` with focused local evidence.

### What is missing

Run the nonduplicated remaining checks: the real Optuna journal smoke; the
CUDA/CosmoLike/deployment-dump saved-state run and `.cpu()` removal control;
the Torch CMB identity and real-CAMB byte-identity checks; and the real-dump
artifact save/rebuild check. Re-run the full board after those inputs are
configured and record any still-current refusal as its own ticket.

<details>
<summary>Technical record for development tools</summary>

- Carries the explicit workstation debt formerly embedded in closed Units 13,
  53, 64/70, 96, and the AI-tree consolidation record.
- Does not duplicate the separately tracked Unit 24, DIDACTICS-62, or Unit 93
  hardware runs.
- The old primary-worktree scratch-fixture drift is not open here; it was
  repaired by the later hygiene change that force-tracks only the disposable
  synthetic backlog fixture.

</details>

<a id="open-schema-v3-gate-fixtures"></a>
## Finish real workstation checks for the current saved-file format

### High-level summary

The repository's temporary saved-emulator examples now contain the scientific
records needed to test loading, reconstruction, and compatibility.

The real cs16 and cs8 CoCoA datasets were generated before those records were
required. Their smoke checks therefore stop during setup and cannot yet test a
real training, save, and prediction run.

Regenerating those datasets is necessary evidence that the current file format
also works with the configured scientific software and data, not only with the
small local examples.

### Current status

**Ticket type: BUG FIX.**

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**OPEN — PARTIAL FIX.** Commit `0fe2067` updated the repository fixtures and
the local CPU save-and-rebuild checks pass. The required real CoCoA smoke runs
have not passed because the cs16 and cs8 datasets do not yet contain their
generator-authored `.facts.yaml` files.

**Severity: MEDIUM.** The stale examples block normal validation work, but the
fault is in development fixtures rather than the production saved-emulator
reader.

### What is already fixed

Commit `d3b9289` authenticates fixed scientific facts through the production
artifact and Cobaya adapter path. Commit `0fe2067` carries the real training
record through the CMB, background, matter-power, and scalar smoke writers;
gives the synthetic geometry fixture a declared support box; and makes the
identity fixtures self-contained. The geometry, scalar, background, CMB,
transfer, and cosmic-shear adapter checks pass locally.

### What is missing

Regenerate the real cs16 and cs8 training and validation dumps so their
producer-authored facts files exist. Then run `geo-paths` with CoCoA and run
the four emulator smoke gates on the configured workstation. Do not replace
those real records with synthetic facts merely to make preflight pass.

<details>
<summary>Technical record for development tools</summary>

- The repository fixture boundary is `ai/gates/checks/geo_paths.py`,
  `scalar_smoke.py`, `cmb_smoke.py`, `bsn_smoke.py`, and `mps_smoke.py`; commit
  `0fe2067` completes that code change.
- The deployment manifest now names the two required facts files beside the
  cs16 and cs8 parameter tables. Both are absent in the current CoCoA checkout,
  so preflight correctly refuses instead of fabricating them.
- Required evidence is a CoCoA `geo-paths` run plus the four real workstation
  smoke gates. Do not close this ticket from local CPU examples or a dry run.

</details>

<a id="open-getdist-column"></a>
## Write the GetDist posterior column with the correct meaning

### High-level summary

The generated chain table is read by GetDist, which expects its second column
to contain the negative logarithm of the posterior probability.

The generator instead labels that column `lnp` and writes the ordinary log
posterior with the opposite sign.

Downstream analysis can therefore reverse which of two samples has the better
posterior value and draw the wrong conclusion from a valid chain.

### Current status

**Ticket type: BUG FIX.**

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**OPEN.** The permanent data rule requires `minus_logpost`; the source and
component README still describe and write the opposite sign.

**Severity: MEDIUM.** A normal GetDist plot or ranking can be misleading, but
the sign does not change generated physics vectors, training, or values served
by an emulator.

### What is already fixed

The generator writes weights, sampled coordinates, and `chi2*` in a stable
table with parameter sidecars.

### What is missing

Write and name negative log posterior, or choose and document a different
table format. Give uniform samples an honest neutral value that does not claim
a posterior evaluation.

<details><summary>Technical record for development tools</summary>
The wrong sign does not alter the generated physics vectors, the training
calculation, or the emulator result served to Cobaya. Owner:
`compute_data_vectors/generator_core.py`. A two-row GetDist known answer must
rank the larger posterior as better; reversing the sign must fail.
</details>

<a id="open-generator-failure-retry"></a>
## Retry failed generator rows reproducibly

### High-level summary

Once the generator reliably refuses to publish a failed row, it can offer a
convenient way to replace that row without changing the random sequence in an
unexplained way.

No complete replacement policy exists yet. A retry could otherwise consume a
different number of random draws on different MPI workers and make two runs
with the same seed disagree.

### Current status

**Ticket type: NEW FUNCTIONALITY.**

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**OPEN.** Resume (`--loadchk 1`) already recomputes rows still flagged
failed without consuming new random draws; this ticket is about drawing
replacement parameter rows instead.

**Severity: MEDIUM.** Automatic reproducible replacement is useful during
ordinary long runs, but a safe program may instead stop without publishing.
The High safety requirement does not depend on automatic retry.

### What is already fixed

Row failures have explicit metadata, and accepted payloads have shape and
finite-value checks.

### What is missing

Define which random state advances after a failed calculation, how MPI
workers receive replacement work, and how the failfile records retries. Prove
that serial and MPI runs follow the documented policy for the same seed.

<details><summary>Technical record for development tools</summary>
Owner: serial and MPI scheduling in
`compute_data_vectors/generator_core.py`. The High failed-row safety repair is
closed and must remain unchanged by automatic retry.
</details>

<a id="open-physics-implementation-identity"></a>
## Record which physics formulas produced each dataset and trained emulator

### High-level summary

Dataset and emulator fingerprints should record which scientific formulas
created every target and analytic base.

The current records omit some formula identifiers, especially for Syren and
for behavior supplied by CAMB or CosmoLike.

A formula can therefore change while an old dataset or saved emulator still
appears compatible, allowing different science under the same identity.

### Current status

**Ticket type: NEW FUNCTIONALITY.**

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**OPEN.** Git and descriptive provenance exist, but semantic or content
identifiers for the physics implementations are incomplete.

**Severity: MEDIUM.** Stable formula fingerprints would make compatibility
checks stronger, but the audit found no current formula collision that has
already produced wrong results. This is a new identity capability, not a
demonstrated High-severity defect.

### What is already fixed

Dataset requests bind family, product, variant, settings, parameter order,
and random-engine policy. Artifacts bind fixed facts and model recipes.

### What is missing

Give every target-producing physics formula a stable semantic or content
identifier in request identity. Give every model output decoder and analytic
base its own identifier in artifact identity, then verify the complete chain
before serving.

<details><summary>Technical record for development tools</summary>
Severity: MEDIUM NEW FUNCTIONALITY; it prevents future wrong science under an
apparently compatible identity. Owners:
dataset request/manifest, fixed facts, artifact compatibility, and
`emulator/syren_base.py`. A semantic formula change changes identity or
refuses; an unrelated documentation commit does not.
</details>

<a id="open-pce-domain-enforcement"></a>
## Refuse polynomial-emulator requests outside the fitted parameter range

### High-level summary

A fitted polynomial has saved lower and upper parameter bounds. A future
strict serving option should reject a point outside that box instead of moving
the point silently to the nearest boundary.

The current forward path clamps distant values. This behavior is known and
does not hide whether the fit itself passed its accuracy test, but it does not
provide the stricter interface wanted for scientific serving.

### Current status

**Ticket type: NEW FUNCTIONALITY.**

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**OPEN.** The separate High ticket first prevents an inaccurate fit from
being saved at all.

**Severity: MEDIUM.** Strict domain refusal is an important serving
capability, but the audit did not find evidence that the existing documented
clamping behavior broadly breaks current PCE use.

### What is already fixed

PCE artifacts save fitted bounds, modes, coefficients, and reported
leave-one-out values.

### What is missing

Add an explicit strict domain contract to `PCEEmulator.forward`, retain only a
small named roundoff tolerance, and add far-out and just-inside examples that
make the behavior clear.

<details><summary>Technical record for development tools</summary>
Owner: `emulator/designs/pce.py::PCEEmulator.forward` and artifact rebuild.
Required evidence includes two far-out points that previously clamped to the
same boundary and tolerance checks at every saved bound.
</details>

<a id="open-power-zero-gradient"></a>
## Preserve the power activation gradient at zero

### High-level summary

The power activation is meant to pass a useful gradient through an input that
is exactly zero so a newly initialized correction can begin learning.

Its current `sign(x) * f(abs(x))` formula returns the right forward value but
automatic differentiation gives a zero derivative at that point.

Zero-initialized layers and padded coordinates can therefore remain unable to
learn even though ordinary prediction checks look correct.

### Current status

**Ticket type: BUG FIX.**

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**CLOSED.** Both production activation classes now compute the signed power
transform as `x` times an even magnitude ratio with analytic limit one at
zero: the direct quotient away from the origin (matching the previous tail
values to rounding) and a justified quadratic series below `|x| = 1e-3`,
with safe substituted inputs so no unselected branch can poison a gradient.
Constructors validate finite positive `p_min < p_max` before any forward
pass. Because the origin derivative is now exactly one, the power families
left the zero-derivative head-refusal set: a power head pin is accepted, and
a frozen-trunk step moves power-activated CNN and transformer heads.

### What is already fixed

Everything in this ticket: the formula, the bound validation, the origin
tests, and the head-liveness consequences.

### What is missing

Nothing for this ticket. The GPU acceptance leg from the permanent model
note remains owed with the other workstation runs.

<details><summary>Technical record for development tools</summary>
Owners: `emulator/activations.py::signed_power_transform`,
`require_power_bounds`, `PowerGatedActivation`, `GatedPowerActivation`;
`ZERO_DERIVATIVE_HEAD_ACTIVATIONS` reduced to `relu`, with the matching
refusal text in `emulator/experiment.py::validate_active_model_values`.
Evidence: `ai/tests/test_power_activation_origin.py` (exact-zero derivative
one for five exponents — the assertion a restored `sign(x) * f(|x|)`
mutation fails — p=1 series-region bitwise identity, tail parity at
rtol 1e-12, seam agreement, float64 gradcheck through zero, bound refusals,
zero-initialized-layer first gradient) and
`ai/tests/test_active_model_validation.py` (power head pins accepted;
frozen-trunk step moves ResCNN and ResTRF power heads; ReLU refusals
unchanged). Whole-model CPU forward with gated_power everywhere measured
1.15x the sign form; the delta is the guarded singular point's extra
elementwise branches.
</details>

<a id="open-adapter-contracts"></a>
## Make every Cobaya bridge check inputs and protect cached results

### High-level summary

Each Cobaya bridge should validate the request, declare the quantities it
needs, and return a result that the caller can use without changing the
bridge's saved state.

The five bridges currently differ in these checks, and several getters expose
arrays backed directly by an internal cache.

One request can therefore be routed with the wrong segment or one caller can
mutate the scientific result that a later caller receives.

### Current status

**Ticket type: BUG FIX.**

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**CLOSED.** Commits `d146590` and `5e0792a` give all five adapters one strict
input and path contract, validate their family-specific request and artifact
facts, publish scalar results through Cobaya's derived mapping, and return
owned public arrays and containers. The follow-up binds the gate to the exact
adapter source surface and corrects its evidence claims.

### What is already fixed

The shared predictor continues to own model evaluation and each adapter keeps
its family-specific scientific transformation. The completed boundary checks
now cover the common and family-specific responsibilities around that core.

### What is missing

Nothing for this ticket. The focused adapter suite ran 25 checks, both
`adapter-contracts` evidence groups passed, the scalar and matter-power
identity gates passed, the board self-test passed, and all 635 CPU regression
tests passed.

<details><summary>Technical record for development tools</summary>
Severity: HIGH; normal Cobaya use can route a wrong segment or return a
mutated cached value, silently changing the scientific result. Owners: all
`cobaya_theory/emul_*.py` modules and shared inference helpers. Live Cobaya
dependency, swapped-segment, strict-type, and mutate-then-read witnesses are
required.
</details>

<a id="open-cmb-covariance-transaction"></a>
## Publish CMB covariance files without overwriting a good result

### High-level summary

The CMB covariance program produces an expensive scientific matrix. A rerun
must not destroy an earlier valid result, and another program must never see a
half-written archive at the filename used for later calculations.

The command now writes a hidden file and closes it before one non-overwriting
hard link gives the completed archive its final name. An existing file or link
keeps that name unchanged, including one created while the calculation is
running.

### Current status

**Ticket type: BUG FIX.**

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**CLOSED.** Commit `4e4e09f` replaces the earlier publication framework with
one short private-write and non-overwriting-link path. It also stops an
occupied output name before YAML parsing or CAMB work.

### What is already fixed

The producer still assembles a complete in-memory member set and checks finite
arrays before saving. It writes beside the destination, closes the archive,
and creates the final name without overwriting a late competing file. The
private name is removed after ordinary success and handled write,
interruption, or link failures.

### What is missing

Nothing for this ticket. Seven focused publication checks, the CMB parameter
witness, the validation-board self-test, and the 91 focused policy and
regression tests pass. An uncatchable process termination may leave harmless
hidden disk debris; that separate remainder is parked below Low.

<details><summary>Technical record for development tools</summary>
Severity: HIGH; a rerun or interruption could destroy a preceding valid
covariance file. Owner: `compute_data_vectors/compute_cmb_covariance.py`.
The gate claim `cmb-covariance-publication.transactional-output` injects
write, final-name, interruption, and late-racer faults. A preceding archive
remains byte-identical and readable.
</details>

<a id="open-training-selection-record"></a>
## Record which saved weights the training run chose

### High-level summary

During training, several candidate weight sets may be compared, including the
untouched model, ordinary epoch snapshots, and moving-average snapshots, so the
saved record must say exactly which one became the published emulator.

The training loop returns histories but no single validated selection record,
so each driver reconstructs the winner afterward and can incorrectly name a
trained epoch or attach the wrong statistics when the untouched model wins.

The user can then receive one emulator file and a scientific report describing
another candidate, even though both parts look internally reasonable.

### Current status

**Ticket type: BUG FIX.**

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**CLOSED.** `training_loop_batched` returns a selection record beside the
histories (candidate baseline or trained epoch, pass-local epoch, raw or
EMA weights, and the winner's fractions, median, and mean), `run_emulator`
stores each pass's record in the resolved recipe and publishes one
run-level `resolved_train["selection"]` (pass identity,
concatenated-history epoch, threshold vector and selection index), and
`validate_thresholds` performs the one-time shape / finiteness / boolean /
strict-order check. The two train drivers, the shared tune objective, and
the saved h5 attributes read the record; no consumer reconstructs a winner
from the histories, so a baseline win is now reported as exactly that.
This implements the design the permanent `training-stack.md` "Selection
record" section already specifies.

**Severity: MEDIUM.** Current evidence shows that the saved report can name
the wrong candidate or statistics, but it does not show that the selected
weight bytes or the emulator's served prediction are wrong.

### What is already fixed

Everything the ticket demanded: the record leaves the training loop as one
validated object, the resolved recipe persists it, and every consumer reads
it. Witnesses: `ai/tests/test_training_selection_record.py`
(baseline-selected, trained-selected, malformed-threshold, YAML round trip)
plus the run-level mapping assertions in
`ai/tests/test_training_pass_recipe.py`; the full suite passes with the
change (813 tests OK).

### What is missing

Nothing. The selection record is returned, validated once, persisted, and
consumed everywhere the histories were previously rescanned.

<details><summary>Technical record for development tools</summary>
The published model and reported evidence can disagree. Owners:
`emulator/training.py::run_emulator`, `EmulatorExperiment.train`, all
drivers, tuning objectives, and artifact publication. Baseline-selected,
trained-selected, malformed-threshold, and round-trip witnesses are required.
</details>

<a id="open-cmb-serving-conversions"></a>
## Add advertised CMB unit and multipole conversions

### High-level summary

The CMB bridge should support the standard unit and multipole-factor choices
that its public documentation advertises. At present, the safe raw stored-unit
path works, but several documented converted requests are refused.

This is separate from accepting physically impossible spectra: refusing an
unsupported conversion is inconvenient but safer than returning a wrong
conversion.

### Current status

**Ticket type: NEW FUNCTIONALITY.**

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**OPEN.** Conversion behavior and its capability report are not implemented.

**Severity: MEDIUM.** The missing conversions contradict advertised
capabilities during normal use, but the current path refuses rather than
silently fabricating a converted spectrum.

### What is already fixed

Artifacts bind spectrum and exact multipole axes, and raw stored-unit serving
has explicit refusal checks.

### What is missing

Implement unit conversion and spectrum-specific multipole factors using the
artifact's fixed `TCMB`, or the exact current value when `TCMB` is sampled.
Make `must_provide` and `get_Cl` report the same capability, and replace the
raw-only refusal checks in the same reviewed change.

<details><summary>Technical record for development tools</summary>
Owner: `cobaya_theory/emul_cmb.py`. Live request, conversion, fixed-versus-
sampled `TCMB`, and capability-agreement witnesses are required. This feature
must preserve the physical validation in `open-cmb-serving-contract`.
</details>

<a id="open-optimizer-scheduler-protocol"></a>
## Reject unsupported training options before a run starts

### High-level summary

The training configuration should describe one optimization procedure that the
chosen device can execute, including when the learning rate changes and which
measurement controls that change.

CUDA setup currently forces a faster optimizer shortcut without proving that
the chosen optimizer supports it, and Apple half-precision training can start
without the protection that prevents very small gradients from disappearing.

These ordinary device or optimizer choices can fail after an expensive run has
started or silently use a procedure different from the one saved in the run
record.

### Current status

**Ticket type: BUG FIX.**

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**CLOSED.** Every named capability is now checked before construction or
resolution, where the user can still fix the configuration. CUDA forces the
fused optimizer implementation only when the optimizer class's constructor
accepts a `fused` argument; an explicitly configured `fused` on a class
without one is refused by name. LBFGS is refused because the loop steps with
no closure. The two per-batch scheduler classes (`OneCycleLR`, `CyclicLR`)
are refused because the loop advances its scheduler once per epoch after
warmup, which would silently stretch a per-batch schedule. Reduced precision
on MPS is refused: MPS autocast runs in float16, whose small gradients
underflow without gradient scaling, and no scaling policy is implemented.

Persisting a scheduler cadence field in the resolved record was judged
unnecessary and is deliberately not done: the cadence is a code-owned
constant, not a configuration choice, and the per-batch refusal removes the
one way a run could follow a different cadence than the record implies. The
plateau scheduler already advances only after its named event (the
post-warmup epoch's validation median).

### What is already fixed

Everything actionable in this ticket, as described above.

### What is missing

Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Owners: `emulator/training.py::_effective_optimizer_extras` (fused
capability by constructor signature), `make_optimizer` (LBFGS closure
refusal), `make_scheduler` (per-batch class refusal), and the new
`resolve_amp_policy` (MPS float16 refusal; bfloat16 elsewhere; policy
"unscaled"). Evidence: `OptimizerSchedulerProtocolTests` in
`ai/tests/test_training_pass_recipe.py` — fused forced on AdamW under CUDA,
absent for Rprop, explicit fused on Rprop refused, LBFGS refused, both
per-batch schedulers refused, MPS use_amp refused while MPS full precision
and CPU bfloat16 resolve unchanged. The pre-existing CUDA recipe test still
records the forced fused value.
</details>

<a id="open-memory-planner"></a>
## Measure memory without changing the model and reserve capacity before allocation

### High-level summary

A memory planner should estimate the complete live cost of a training point
without changing the model, the random-number generators, or the data that the
real run will use.

Current sizing probes can alter model or random state, omit buffers and
mixed-precision copies from their count, and let a worker allocate memory
before it has reserved capacity.

A parameter study can therefore produce different results merely because it
measured memory, or run out of memory before the protection mechanism has a
chance to stop the allocation.

### Current status

**Ticket type: BUG FIX.**

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**CLOSED — the described repair is not worth building.** An audit of the
current sizing path found no demonstrated failure behind any of the three
requested changes:

- The batch-term probe runs one dummy forward on zeros through the live
  model with scoped saved-tensor hooks. The current model families carry no
  batch-normalization running statistics and no active dropout, so that
  forward changes no model state and consumes no random numbers; the graph
  is discarded. A state-preserving rework would guard against module types
  the library does not use.
- The parameter budget already multiplies weight bytes by five (weights,
  gradients, and a three-slot optimizer worst case), and the probe measures
  the real autograd-saved activations. The omitted terms — index buffers,
  bound vectors — are kilobytes against that padded budget. Finer
  accounting would add estimate complexity without a failure it prevents.
- A capacity-token reservation before worker allocation is concurrency
  machinery whose only benefit is converting a visible out-of-memory
  failure into a queue wait. Sweeps already size each source against the
  device's reported free memory and refuse a budget that cannot hold
  resident state plus one complete batch.

The one real remainder — a future model family with stateful-forward
modules would make the probe mutate state — is below Low and parked as
[Guard the sizing probe if a stateful-forward family is added](#parked-memory-probe-stateful-forward).

### What is already fixed

Batch sizing accounts for packed target bytes, measures real saved
activations, and refuses a budget that cannot hold one complete batch.

### What is missing

Nothing. The described extension is declined as disproportionate to any
demonstrated failure.

<details><summary>Technical record for development tools</summary>
Audit surface: `emulator/batching.py::compute_batch_byte_terms`,
`compute_model_size_bytes`, `batches_per_load`, and
`emulator/scheduling.py::estimate_train_vram_fraction`. No
`register_buffer` site in the model designs stores more than index or bound
vectors; no batch-normalization or dropout module appears in any design.
</details>

<a id="open-resolved-run-record"></a>
## Save every effective setting and reset each repeated study

### High-level summary

A resolved-run record should contain the settings the program actually used,
after defaults and command-line choices were combined, so the same training
point can be reproduced later.

Several effective values are still omitted, and repeated sweep or tuning
points can reuse an experiment after an earlier point changed its weights,
random state, or settings.

A reported result may therefore be impossible to reproduce, and two sibling
study points may not begin from the same fair starting state.

### Current status

**Ticket type: BUG FIX.**

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**OPEN.** Resolved YAML and core study manifests exist; run totality, pristine
source identity, and root-configuration validation are incomplete.

### What is already fixed

Model, optimizer, several loss and schedule values, and composition facts are
saved in current results.

### What is missing

Persist effective rows/tails, update horizons, scheduler protocol, selection,
and pristine study identity. Rebuild each repeated point from one authenticated
source state and validate the complete configuration tree with close-match
errors.

<details><summary>Technical record for development tools</summary>
Severity: MEDIUM with reproducibility impact. Owners: training resolver,
experiment configuration, tune/sweep drivers, and artifacts. Reordered points
must remain identical; unknown/misnested config and a state mutation must fail.
</details>

<a id="open-study-diagnostics"></a>
## Publish structured study and diagnostic results

### High-level summary

A study should distinguish a successful result from a failed or unavailable
point, and its tables and plots should be created only from complete compatible
scientific values.

Current workers can turn failures into ordinary rows filled with NaN values,
while diagnostic and plotting helpers can accept empty, nonfinite, truncated,
or mutually incompatible inputs.

The finished study can therefore look complete while hiding failed work, or
show a visually convincing comparison whose rows and curves do not represent
the same scientific cases.

### Current status

**Ticket type: BUG FIX.**

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**OPEN.** Core study manifests and shared diagnostic-domain screening exist;
point status, cleanup, memory bounds, and final publication validation are
partial.

**Severity: MEDIUM.** A normal study can hide failures or compare mismatched
cases, but the demonstrated endpoint is a diagnostic table or plot. Current
evidence does not show changed training data, saved weights, or a value served
by the emulator.

### What is already fixed

Several diagnostic quantities have independent formulas and saved plots or
tables, and known families share a result writer.

### What is missing

Use structured success/failure/unavailable point results, clean sibling
processes, refuse nonfinite or empty diagnostic publication, bound wide-array
memory, validate complete table lengths, and choose plot scales/colors from
validated data rather than forcing them.

<details><summary>Technical record for development tools</summary>
A normal study can hide failures or present mismatched cases as a scientific
comparison. Owners: sweep/tune workers,
`emulator/diagnostics.py`, `results.py`, and
`plotting.py`. Failure-row, cleanup, empty/nonfinite, truncation, log-scale,
and color-identity witnesses are required.
</details>

<a id="open-mps-request-contract"></a>
## Validate matter-power requests before a run starts

### High-level summary

Cobaya tells the matter-power bridge which products a likelihood will use
before sampling starts. A clear early check should refuse an unsupported
particle pair, nonlinear setting, redshift request, or wavenumber request at
that point, while the user can still correct the YAML.

The current `must_provide` method notices only the optional sigma-eight
quantity. Other malformed requests may therefore survive setup and fail later
when a likelihood calls a getter. The existing getters and serving-range
checks prevent the unsupported request from becoming a published spectrum,
so this is an early-error and clarity problem rather than demonstrated wrong
science.

### Current status

**Ticket type: BUG FIX.**

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**CLOSED. Severity: MEDIUM.** `must_provide` now validates every `Pk_grid`
and `Pk_interpolator` requirement while Cobaya setup can still stop: only the
`delta_tot` density pair is accepted, the nonlinear choice must be boolean,
requested redshifts must lie inside the stored z grid (z is never
extrapolated), and a requested `k_max` must be servable — inside the stored
grid for the raw grid, and beyond it for the interpolator only when the
power-law tails are enabled. Each refusal names the observed request, the
stored bound, and the corrective action. The caller's requirement mapping is
only read, and the sigma8-to-H0 behavior is unchanged.

### What is already fixed

Everything: the early request validation plus the previously completed getter
and serving-range checks.

### What is missing

Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Owner: `cobaya_theory/emul_mps.py::must_provide` with the new
`_validate_matter_power_request` helper. Evidence: the
`MatterPowerRequestContractTests` group in
`ai/tests/mps_sigma8_child_checks.py` — accepted in-range requests (with
sigma8 still adding only H0), refused wrong pair, out-of-range z above and
below, grid `k_max` beyond the stored edge, interpolator `k_max` obeying
`allow_k_extrapolation`, non-mapping options, and non-boolean nonlinear. The
live-Cobaya routing test still passes with bare `None` requirements.
</details>

<a id="open-implementer-blocked-outcome"></a>
## Let the Implementer stop honestly when a ticket cannot proceed

### High-level summary

An Implementer can discover that a ticket needs an Architect decision, missing
hardware, a corrected acceptance test, or permission to touch another file.
Without an accepted way to report that situation, a capable model may keep
editing speculatively, weaken a restriction, or describe activity as progress
because it believes that stopping means failure.

Generalize the existing blocked handoff into one legitimate, structured
outcome. `BLOCKED` means that the Implementer stopped safely and supplied the
evidence needed for an Architect decision. It is not a failed candidate and
must never trigger an automatic instruction to try harder.

### Current status

**Ticket type: NEW FUNCTIONALITY.**

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**CLOSED — the honest stop exists as the checkpoint family, and the enum on
top of it is declined.** The Implementer role protocol requires a relayable
`IMPLEMENTER_HANDOFF` block for every stop, including hitting a blocker: a
mid-unit stop is titled `CHECKPOINT` and carries the changed files, the
completed checks, the unfinished work, and the decision requested from the
Architect. The specialized stops each have their exact validated shape —
`BUDGET BLOCKED` with the over-limit measurement, the digest-bound
capability checkpoint for a rejected subagent launch, and the
`CONTEXT HANDOFF` for a replaced context. The daemon accepts the checkpoint
headings, routes the return to an Architect checkpoint audit instead of
retrying, preserves active Implementer work and saved checkpoints across a
restart, and a checkpoint commit is never candidate C. What remains of the
request is a five-value blocker-reason vocabulary with parser validation
and per-reason tests. That vocabulary has no mechanical consumer: the
daemon must treat every reason identically — route to the Architect, never
retry, never manufacture a candidate — and the Architect reads the required
free-text evidence regardless of which label sits above it. A taxonomy
whose only reader already reads the evidence is parser surface without a
failure it prevents.

### What is already fixed

The Implementer must pause after 90 minutes, may return a blocked subagent
checkpoint, and cannot turn either checkpoint into Architect GO or a landing.
The Architect can revise the same ticket after inspecting saved evidence.

### What is missing

Nothing. The checkpoint family in `.claude/OPUS_ROLE.md` and the daemon's
checkpoint routing carry every load-bearing property the ticket asked for;
the blocker-reason vocabulary is declined as validation machinery without a
mechanical consumer.

<a id="open-stale-landing-reaudit"></a>
## Recover safely when main advances after a landing is prepared

### High-level summary

After the Architect accepts a ticket, the watcher prepares an exact landing
commit against the current `main`. For example, another user action may add a
legitimate commit to `main` before that prepared landing is installed. The
watcher correctly refuses to apply the old prepared result, but it does not
yet provide a supported way to request a fresh Architect audit against the new
parent and continue the same ticket.

This is a safe stop rather than a wrong merge: the candidate, prepared
landing, GO request, and user files remain preserved. The missing recovery can
still interrupt a normal maintenance session and require Git expertise, so a
future watcher command must make that recovery explicit and repeatable.

### Current status

**Ticket type: BUG FIX.**

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**CLOSED — the supported recovery is the ordinary path, and a shortcut
would weaken it.** When `main` moves under a prepared landing, the watcher
refuses, preserves the candidate and prepared commit, and exits nonzero;
the ticket that cycle served is still open in this backlog, so restarting
the watch runs a fresh cycle against the new `main`. The Implementer redoes
the work on the actual new parent and the Architect audits the real final
candidate under the same uniform rule as every other landing. The cost of
that fallback is one repeated Implementer turn for a rare event — a user
advancing `main` in the middle of a watch. The requested alternative —
stale-marking, a recomputed provisional integration of the old candidate
onto the new parent, a bounded re-audit protocol, replacement-landing
binding, and real-Git witnesses for each scenario — is a second acceptance
route through the daemon's highest-trust code, in which the Architect
reviews only the interaction between the old GO and the intervening
commits instead of a complete candidate. Maintaining a permanently weaker
route to save one occasional turn is a bad trade.

### What is already fixed

The landing commit is saved on a private exact Git reference. A dirty or moved
user checkout is not reset, cleaned, overwritten, or silently merged.

### What is missing

Nothing. Restarting the watch on the still-open ticket is the supported
recovery; it keeps one uniform full-audit landing rule instead of adding a
focused-revalidation shortcut.

<a id="open-relay-log-identity"></a>
## Give every role run its own relay-log filename

### High-level summary

Each role run saves its terminal output under `ai/notes/relay/` so the
Architect can inspect what actually happened. The filename currently uses the
role name and a timestamp with only one-second precision.

Two quick runs of the same role inside one second can therefore choose the
same path. The later run may replace the earlier evidence, leaving only one
log even though two mailbox messages were processed.

### Current status

**Ticket type: BUG FIX.**

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**CLOSED.** The dispatch log path comes from an exclusive-creation
reservation: the operating system refuses an existing name, and a two-digit
suffix is appended until a fresh name is accepted, so a second turn of the
same role inside one clock second keeps its own complete log instead of
truncating the first. The readable timestamp and role name stay in the
filename. `ai/tests/test_relay_log_reservation.py` hands the reservation
one frozen stamp — no clock mocking — and requires both same-second logs to
survive with their own contents, the exact suffix order, no suffix across
different roles, and creation of a missing relay folder. The
`handoff_router` transport-copy path already reserved its sequence
atomically; the dispatch path uses the same reservation idea at its own
boundary.

### What is already fixed

Role messages themselves have unique sequence numbers and their mailbox
archive does not depend on the relay-log timestamp.

### What is missing

Nothing. `ai/tools/mailbox_dispatch.py::reserve_dispatch_log_path` owns the
reservation, and the frozen-stamp witness covers the same-second collision.

<a id="open-python-prose-review"></a>
## Make tracked explanations describe one coherent current library

### High-level summary

Tracked explanations should make the repository look like one deliberately
designed current library. Some READMEs, permanent notes, comments, docstrings,
help text, and diagnostics instead preserve dated “hard user rule” labels, old
bug-report names, review waves, or later corrections beside earlier rules.
That patch-by-patch narration makes a human reconstruct which paragraph is the
real rule.

Function explanations have a second widespread problem. Most functions in
`ai/tools/` and `ai/tests/` do not begin with a docstring that tells a new
reader what the function does, how it does that job, what each input means, and
what comes back. A short label or an old ticket name is not enough for a
student or future developer to understand the function without reverse
engineering its body.

Review the complete tracked repository after the existing backlog is closed.
Rewrite policy chronology as the current behavior, its technical reason, and
the reader's action. Give extra attention to `ai/` and
`compute_data_vectors/`, where explanatory Python prose has often depended on
old ticket labels. Keep a date only when it is part of the subject itself,
such as a scientific release or citation, and record why it must remain.

### Current status

**Ticket type: BUG FIX.**

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**OPEN.** The function-docstring portion is underway at the user's explicit
order, one file per commit, with each commit proving the change is
docstring-only by comparing the two versions' abstract syntax trees with
docstrings stripped and by a full green test-suite run. Completed so far in
`ai/tools/`: `role_contract.py`, `backlog_guard.py`, and
`permanent_note_guard.py`. The user then redirected the sweep to the
emulator package, where docstrings carried the required structure but
assumed domain and Python knowledge (a sidecar, the bool-int subclass
trap, struct round trips, Simpson's rule, sigmoid, broadcasting) without
defining it. The emulator-package arm is COMPLETE: fourteen files were
rewritten (`parameter_table.py`, `validation.py`, `cocoa.py`,
`background.py`, `family_drivers.py`, `analytics.py`, `syren_base.py`,
`scheduling.py`, `model_recipe.py`, `activations.py`, `batching.py`,
`diagnostics.py`, `data_staging.py`, `plotting.py`), and the remainder
was audited function by function and found already at the required
depth with nothing to change (`inference.py`, `warmstart.py`,
`fixed_facts.py`, `results.py`, `training.py`, `experiment.py`, and
the designs/geometries/losses subpackages, whose nn.Module classes
teach through class docstrings with forward-Arguments blocks).
The `compute_data_vectors/` arm is COMPLETE: four files rewritten
(`generator_core.py` and the background / cmb / mps drivers, whose
store overrides now teach the memmap modes, the shared RAM decision,
and the atomic save), with `dataset_generator_lensing.py` and
`compute_cmb_covariance.py` audited clean. The `cobaya_theory/` arm is
COMPLETE: all five adapters now teach the Cobaya lifecycle (who calls
initialize and what extra_args is), the closed-option-list reason, and
(emul_mps) the bicubic spline; their module and class docstrings were
audited clean. The `ai/tools/` arm is underway: twelve more files are
complete (`candidate_admission.py`, `review_dispatch.py`,
`implementer_checkpoint_hook.py`, `reopen_transition.py`,
`control_plane_handoff.py`, `provider_health.py`,
`mailbox_providers.py`, `ticket_change_guard.py`, `mailbox_watch.py`,
`mailbox_tickets.py`, `mailbox_store.py`, `backlog_bundle.py`),
joining the three swept earlier. The `ai/tools/` arm is COMPLETE
at the strict mechanical bar: after a user spot check found
surviving one-line docstrings and missing Arguments blocks (for
example `consume_daemon_message` and the `recover_failed_*` family
in `mailbox_recovery.py`), a stricter census — every parametered
callable must carry an Arguments block, every value-returning
callable a Returns block, and no non-trivial body may keep a
one-line docstring — found 153 further weak docstrings across 17
files, and a 17-commit remediation drove that census to zero
findings over all twenty-five files, including the ten large part
files (`mailbox_daemon.py`, `mailbox_dispatch.py`,
`mailbox_recovery.py`, `handoff_router.py`,
`mailbox_control_plane.py`, `mailbox_cycles.py`,
`handoff_contract.py`, `mailbox_envelopes.py`,
`mailbox_landing.py`, `mailbox_worktrees.py`).
A later user spot check caught undefined jargon surviving in the
completed emulator arm: `family_drivers.py` used "sweep",
"N-train sweep", and "sweep point" without defining them, and its
docstrings now define that vocabulary in plain words (one commit,
same stripped-AST and full-suite proofs). The user then judged
`results.py` "quite dry" and ordered a pass over every emulator
file; the strict census found 50 more weak docstrings across nine
files (undocumented `__init__` / `forward` methods of the design
classes, factory and walker closures), all fixed, and a
whole-package re-read removed the remaining review-history
narration ("unit N" ticket citations, "byte-identical to before",
removed-variant pointers at git history) and defined the surviving
terms of art in place (artifact, HDF5, schema, provenance,
composition, materialize, the finite contract, the norm/act call
shapes). Two refusal strings that told the user about an "inert
width contract" / "the contract the dataset was generated under"
were reworded in two string-only commits; the "finite contract"
and "chi2 domain contract" error texts are asserted verbatim by
tests and the finite-contract gate and keep their names, now
defined in prose where they are raised. The emulator package is
census-zero over all forty files, every commit stripped-AST-proven
(or diff-limited to the named strings) with the 813-test suite
green. A further user reading of `training.py` flagged three more
undefined terms of art -- the Anchor class ("decoupled L2-SP
anchor"), the phase-boundary "fingerprint"/"digest"/"witness"
family, and a "sqrt-rule batch anchor" phrase colliding with the
weight-anchor sense -- all now defined in place (the anchor as a
spring toward the starting weights, L2-SP expanded, the SHA-256
digest and witness line explained, the bs_base refusal reworded in
a proven single-string change). A user-ordered coverage review of
`documentation/emulator_code_guide.tex` then found and fixed five
stale or missing claims (the artifact section denied the pair token,
the artifact tree omitted the scientific-record groups, three
passages inverted the executed anchor/EMA order against the code and
the guide's own listing, the prediction section lacked the
training-region refusal, and L2-SP was unnamed), recompiled and
committed with the PDF; `emulator/README.md` gained the ordered
Appendix D2 reading guide naming each file's one or two main
functions, every symbol grep-verified; the guide also gained a
worked build-a-YAML subsection (the shipped scalar file walked
block by block, the family-selection-by-presence rule, and the
sweep block, every snippet copied from a shipped runnable file).
Remaining under this ticket: `ai/tests/` and the repository-wide
chronology rewrite outside `emulator/`. The
required depth follows the recorded reader standard: no unstated
Python mechanic or term of art, with Arguments, Returns, and Raises
blocks in the aligned name = description form.

**Priority: LOW.** The user explicitly said “after the backlog is closed.”
This improves maintainability and teaching but does not repair a current
scientific result, data-loss path, or halted core operation.

### What is already fixed

The permanent README and Python-style GO/NO-GO notes define the required
human-first voice. They require one coherent current-system account, complete
sentences, concrete examples for abstract ideas, accurate arguments and
returns, and explanations of non-obvious units, shapes, invariants, side
effects, and refusal reasons.

### What is missing

Build a complete inventory of tracked READMEs, the eleven permanent notes,
other tracked developer or scientific documentation, and every tracked
`*.py` file. Review Python module, class, method, and function docstrings,
explanatory comments, command help, diagnostics, and explanatory strings. A
file is not complete merely because it has a docstring. The prose must make
sense to a reader who has never seen the backlog, a Red Team report, an old
ticket, or a development-session label.

Audit every function and method in `ai/tools/` and `ai/tests/` explicitly.
Immediately after its definition, give it a multiline docstring with this
human reading order:

1. One direct sentence naming the function's job.
2. A short plain-language explanation of the important mechanism, including
   why a non-obvious step exists.
3. An `Arguments:` list that explains the meaning of each input, when the
   function has inputs beyond `self` or `cls`.
4. A `Returns:` section that describes the value and its important shape,
   units, or structure, when the function returns a value.
5. A `Raises:` section only when a refusal is part of the interface and the
   caller needs to understand it.

Small test methods and tiny private helpers may be concise, but their
docstrings must still say the concrete behavior they check or perform. Longer
functions need enough explanation to let a reader follow the body line by
line. Do not satisfy this requirement with boilerplate copied between
unrelated functions.

Apply `ai/notes/readme-go-no-go.md` to every covered explanation and
`ai/notes/python-changes-go-no-go.md` to Python prose. Replace labels such as
`DIDACTICS-62`, “Unit 8,” `hard user rule`, wave or round names, development
dates, and ticket anchors with the actual behavior they were standing in for.
Keep a date or historical identifier only when the subject would become false
without it, and explain that necessity at its first use.

Do not rewrite existing Git commits or their messages. Protected history is
immutable. This ticket cleans current tracked files and future commit-message
templates only.

Do not change computational or scientific behavior as part of this ticket.
For a file changed only in comments or docstrings, prove that its
before-and-after syntax trees are identical after docstrings are removed. For
command help, diagnostics, or another explanatory string, require the
executable diff to contain only the intended string literal changes and run
focused exact-output and return-code tests. Run `py_compile` for every changed
Python file and render every changed README. Divide the review into bounded,
non-overlapping batches, but keep one complete inventory so no covered file is
silently skipped.

<details><summary>Technical record for development tools</summary>

Mandatory examples: `ai/gates/checks/d5_training_behaviors.py` must explain
the CPU calculations and training behaviors it protects instead of using
`DIDACTICS-62`. A tracked rule labeled with a development date or `hard user
rule` must become one undated current rule with its technical reason.

The reference form for a non-trivial function is
`compute_batch_byte_terms`: a direct summary, a short mechanism explanation,
an `Arguments:` entry for each input, and a `Returns:` description that tells
the reader what the dictionary contains. The review must report the number of
function and method definitions inspected in `ai/tools/` and `ai/tests/`, the
number that needed changes, and the number left without a docstring. The last
number must be zero before this ticket can close.

Permanent-note findings are returned to the Architect. The Implementer and
Red Team never edit those eleven files. README and Python-prose changes follow
their normal ownership and review paths.

Priority dependency: every ticket listed as Open at admission precedes this
work. The separate widespread `ai/tools/` and `ai/tests/` bug audit may collect
functional defects, while this ticket changes tracked explanations only. If
the prose review reveals a behavioral defect, record a separate bug ticket at
its evidence-based severity instead of repairing it inside this documentation
change.

</details>

<a id="open-candidate-circumvention-review"></a>
## Check an accepted candidate for workarounds around rejected instructions

### High-level summary

An Implementer does not need malicious intent to work around a rejection. A
capable model may preserve its preferred design under another name, weaken a
test so the result passes, or move denied behavior into a wrapper or optional
configuration because that appears to finish the task efficiently.

The Architect already decides whether a change is accepted. Add one short,
explicit review that asks whether the exact candidate obeys both the requested
work and every stated prohibition, without turning the workflow into a
maximum-security system.

### Current status

**Ticket type: NEW FUNCTIONALITY.**

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**CLOSED.** The audit section of `.claude/FABLE_ROLE.md` carries the
consolidated **CIRCUMVENTION CHECK**: five questions answered against the
exact base-to-candidate diff before any GO — prohibition rows preserved
even through generated files and wrappers, no rejected design recreated
under other names, no optional route restoring denied behavior, no checker
change that lets the same candidate pass where the unchanged checker would
object, and no evidence bound to a different commit. The daemon boundary
facts the ticket requested already exist in the admission path: the
candidate must descend from its saved base, changed paths come from a real
`git diff` with non-UTF-8 names refused, undeclared paths surface as scope
findings for the Architect, protected paths refuse outright, and the
character guard walks every raw diff entry with its modes and refuses
submodule entries and uncountable binaries. Hard daemon refusals for
executable bits, symlinks, and large additions are declined: Git prints
every mode and type change in the same raw diff the audit reads, and a
legitimate candidate can contain an executable script, so those cases are
Architect judgment, not admission mechanics.

### What is already fixed

The Architect owns GO and NO-GO, explicit scope, character limits, and the
accepted landing. Required commands run against the exact candidate. The Red
Team is advisory, and neither the Implementer nor the Red Team can approve a
workaround or edit the Architect-owned backlog.

### What is missing

Nothing. The checklist lives in the role file's audit section, and the
admission mechanics named in the status cover the boundary facts; the
declined hard refusals stay Architect judgment over the audited raw diff.

<a id="open-control-plane-protection"></a>
## Protect control files and keep candidates from weakening their own audit

### High-level summary

The files that define role authority and mailbox state must not be changed by
the Implementer. A candidate can also appear to pass by changing the tests,
gate definitions, tolerances, expected output, logging, or exit-code handling
used to judge that same candidate.

Make the first group categorically Architect-only. Continue allowing ordinary
test and tool improvements, but require the Architect to inspect those changes
adversarially and use trusted audit machinery where a candidate could otherwise
weaken its own checker.

### Current status

**Ticket type: BUG FIX.**

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**CLOSED — the enforced boundary exists and is machine-checked.** The
protected set the ticket demands is one complete list in
`ai/notes/role-contract.yaml`: candidates may not touch `CLAUDE.md`,
`.gitattributes`, `.gitignore`, `.gitmodules`, the tracked backlog and its
guard files, or any path under `.claude/`, `.codex/`, `ai/tools/`,
`ai/notes/mailbox/`, or `ai/notes/relay/`, and the contract reader's safety
floor refuses a contract that drops any of those entries. The eleven
permanent notes carry the SHA guard, and the Architect administration turn
is the one legitimate update path, revalidated by the daemon at landing.
For candidate changes to `ai/tests/` or `ai/gates/`, the audit's
gate-integrity screen treats an unnamed change to the gate surface as
tampering and the consolidated circumvention check asks whether a checker
was weakened for its own candidate; protected control-plane candidates
additionally pass the trusted shadow validation and the mandatory
pre-landing review. The one sketched remainder — a separate fingerprint
store with trusted copies of test drivers and tolerance policies — is
declined: the audited base commit in Git is the trusted copy, and a
parallel store would be a second source of truth with its own drift.

### What is already fixed

The Implementer cannot issue its own GO or land its own commit. Permanent notes
and the tracked backlog have separate integrity checks, and exact-candidate tests
already detect several kinds of evidence drift.

### What is missing

Nothing. The contract's forbidden lists, the safety floor that refuses their
removal, the permanent-note guard, the administration path, and the audit's
gate-integrity and circumvention checks are the enforced boundary; the
separate trusted-copy store is declined as a second source of truth.

<a id="open-character-budget-planning"></a>
## Plan a limited ticket across code, documentation, and protected notes

### High-level summary

When `--max` is positive, its character limit applies to the complete final
commit. The Architect therefore cannot spend the whole allowance on Python
and discover later that required explanations, LaTeX documentation, tests, or
Architect-only permanent-note updates no longer fit.

Add an advisory planning reminder rather than a new rejection rule. A useful
starting estimate is 40 percent Python, 50 percent README or LaTeX material,
and 10 percent reserved for possible protected-note work, but the Architect
may choose a different balance for the actual ticket.

### Current status

**Ticket type: NEW FUNCTIONALITY.**

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**CLOSED — the reminder guards a failure that already recovers cleanly.**
When a clean candidate exceeds a positive limit, the Implementer returns the
exact over-limit blocked handoff with its measured count, and the Architect
revises the plan; the cost of discovering a bad allocation late is one
returned turn, not lost work. Estimating the whole landing before writing
the directive is ordinary Architect planning that the directive template
already forces, because the acceptance checklist must require the exact
candidate's guard result to be within the limit. Publishing a starting
proportion as tracked guidance — with a contract test that polices the
wording of that guidance — converts a planning habit into machinery that
must itself be maintained, reviewed, and defended, for no enforcement gain
over the refusal that already exists.

### What is already fixed

Zero disables only the numerical maximum, while positive limits measure code,
tests, documentation, and other changed text together. Readability and required
evidence cannot be sacrificed to meet the number.

### What is missing

Nothing. The advisory proportions are declined as tracked machinery; the
existing over-limit refusal and the blocked-handoff return already make a
bad allocation visible and recoverable at the cost of one turn.

<a id="open-change-risk-classification"></a>
## Use change risk as well as character count when choosing checks

### High-level summary

Character count prevents a maintenance ticket from growing unexpectedly, but
it does not measure consequence. A short numerical-normalization change can be
more dangerous than a much longer documentation and regression-test update.

Keep `--max` as a size guard and add a separate risk label that helps the
Architect choose proportionate validation. Small numerical and scientific-model
changes should receive stronger checks even when their character count is low.

### Current status

**Ticket type: NEW FUNCTIONALITY.**

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**CLOSED — the directive machinery already forces this decision per
ticket.** Choosing validation proportionate to the changed surface is an
existing Architect duty with concrete enforcement points: every changed
Python path must carry a hot or cold classification, every applicable
style-contract row must be copied into the directive with named evidence, a
numerical or scientific change triggers the benchmark and regression rows
of the Python contract, and the Architect selects the acceptance commands
for each ticket individually. A parallel label taxonomy would restate that
duty as vocabulary while adding its own maintenance: labels need stable
meanings, boundary adjudication, and focused examples, all reviewed like
any other tracked guidance. The ticket's own record shows no case where the
size guard was treated as the risk decision, so there is no failure for the
taxonomy to prevent.

### What is already fixed

Documentation and tests count toward `--max`, zero does not waive quality, and
the Architect already chooses acceptance commands for each ticket.

### What is missing

Nothing. The label taxonomy is declined; the hot/cold classification,
per-ticket contract rows, and Architect-selected acceptance commands are
the working form of the same decision.

<a id="open-normalized-implementer-output"></a>
## Normalize untrusted Implementer output before the Architect reads it

### High-level summary

Repository comments, logs, commit messages, test output, and Implementer prose
may contain text that looks like an Architect or system instruction. Even
without malicious intent, placing that raw text beside trusted role policy can
encourage a later agent to treat evidence as authority.

Give the Implementer a small one-way structured report. The daemon validates
and normalizes it, while any retained raw text is clearly bounded and labelled
as untrusted evidence that cannot issue instructions.

### Current status

**Ticket type: NEW FUNCTIONALITY.**

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**CLOSED — the prompt boundary already exists where it decides anything.**
Every dispatched turn places the trusted material first: the daemon's own
banner, the role preamble pointing at the authoritative role file, and the
byte-unchanged common preamble. The raw mailbox body enters only after the
exact `--- MESSAGE ---` delimiter, as the prompt's suffix, so a reader
always knows where evidence begins. Structured validation also exists for
the returns that carry authority: the handoff contract checks envelope,
exact rows, and admission values, and the daemon acts only on parsed exact
headers — instruction-shaped prose inside a message body cannot become an
action, whichever role wrote it. What remains of the proposal is a
versioned report format with Unicode normalization, terminal-escape
scrubbing, and duplicate-field rejection. That is input sanitation
proportioned to hostile bytes, not to a cooperating Implementer whose worst
case is prose that imitates an instruction — prose the Architect role file
already directs the reader to treat as evidence under review, never as
authority.

### What is already fixed

The Architect makes the decision, the Implementer cannot land its own work,
and structured mailbox fields already exist for several handoffs.

### What is missing

Nothing. The trusted-first prompt order, the exact message delimiter, the
handoff contract's structural checks, and header-only daemon parsing cover
the routes where imitated authority could matter; the remaining sanitation
format is declined as disproportionate to a cooperating Implementer.

<a id="open-authenticated-control-messages"></a>
## Authenticate trusted ticket and landing messages proportionally

### High-level summary

The Implementer must not be able to forge an Architect approval, daemon landing
record, Red Team closure, cycle identity, or candidate-admission token. Plain
text fields alone make an accidental or goal-driven imitation harder to
distinguish from a trusted control message.

Add lightweight local authentication rather than a maximum-security system.
The goal is to stop casual forgery and incidental rule-bending, not a hostile
operator with full access to the machine.

### Current status

**Ticket type: NEW FUNCTIONALITY.**

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**CLOSED — the described defense answers a threat this workflow does not
face.** The protection target is an Implementer that bends rules to make its
own ticket pass, not an operator who studies the control plane to counterfeit
credentials. Against the real target, the structural controls already close
every forgery route the ticket names: the daemon acts only on parsed exact
envelope headers, never on free text inside a role's output; an Architect GO
is consumed only from the Architect lane's own outbound and must bind the
exact candidate commit; the daemon alone creates the landing commit, after
the deciding Architect process has exited; and the sealed backlog plus full
commit hashes bind every durable record to exact bytes. Text that merely
imitates an approval therefore has no path to authority, whichever role
prints it. A local secret-key layer on top would not add a boundary — the
key would live on the same machine where every role runs, so any process
able to write a forged message could also read the key — while key creation,
storage, rotation, restart, and recovery would become permanent control-plane
complexity guarding against a forgery class never observed here.

### What is already fixed

The daemon owns landing, the Architect owns decisions, and the Red Team is
advisory. Full commit hashes already identify the candidate and landing in
critical paths.

### What is missing

Nothing. The secret-key layer is declined: it guards against deliberate
credential counterfeiting, which is outside this workflow's protection
target, and it cannot create a real boundary on a single machine where
every role can read the same files the daemon reads.

<a id="open-control-plane-regression-runner"></a>
## Run every required control-plane regression with one command

### High-level summary

The AI test folder contains ordinary `test_*.py` modules and separate
`*_repro.py` programs that rebuild interruptions, stale landings, missing Red
Team operation, worktree failures, push debt, and restart recovery. The
documented `unittest discover` command runs only the first group, so the phrase
“the AI tests passed” can accidentally omit important workflow reproductions.

Add one authoritative command for control-plane acceptance. A user or
Architect should be able to run it once and receive an honest summary stating
which unit tests, contracts, and full workflow reproductions ran and whether
every required check passed.

### Current status

**Ticket type: NEW FUNCTIONALITY.**

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**CLOSED.** `python3 ai/tests/run_control_plane_regressions.py` is the
acceptance command for changes to the AI workflow controller. It runs the
control-plane `test_*.py` modules and every reproduction program from one
explicit manifest, each command in its own child process from the
repository root, prints each command with one verdict line, writes complete
output to a named log file, and returns zero only for a complete pass. It
refuses to start — exit code 2, before any check — when a manifest entry is
missing or duplicated, lacks its README inventory row, or when a
`*_repro.py` file on disk is not listed, so adding a reproduction without
registering it fails loudly instead of quietly narrowing the surface. There
is no skip mechanism: a check that cannot run is a failure. The
trusted-runner escalation this body sketched for protected control-plane
candidates is declined as this ticket's scope: the protected path already
has its own shadow validation, and a convenience command must not become a
second authority over what lands.

### What is already fixed

The folder inventories discoverable tests and direct reproductions separately,
and the protected-ticket path has a D0-owned shadow check for exact candidate
admission and landing. Individual mailbox and recovery reproductions already
return nonzero when their scenario fails.

### What is missing

Nothing. The runner, its manifest checks, and its README documentation are
in the tree; scientific gates and accelerator-bound tests stay in their own
documented acceptance paths so this command needs only a CPU and Git.

<a id="open-daemon-authority-modules"></a>
## Reduce daemon risk through small authority-boundary extractions

### High-level summary

`ai/tools/mailbox_daemon.py` has grown to roughly fourteen thousand lines. Its
recovery protections are valuable, but provider calls, mailbox movement,
candidate records, backlog closure, landing, worktree synchronization, GitHub
push recovery, and restart behavior now interact inside one very large file.

The size is not evidence that the daemon is currently wrong. It is a warning
that future changes will become harder to understand and review unless stable
responsibilities gradually move into smaller modules with clear authority.

### Current status

**Ticket type: NEW FUNCTIONALITY.**

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**CLOSED — the requested extraction exists in the tree.** The daemon is a
coordinator of about two thousand lines beside part files split along the
responsibility boundaries this ticket proposed: provider commands
(`mailbox_providers.py`), dispatch (`mailbox_dispatch.py`), the durable
store (`mailbox_store.py`), message envelopes (`mailbox_envelopes.py`),
ticket cycles (`mailbox_cycles.py`, `mailbox_tickets.py`), landing and push
debt (`mailbox_landing.py`), recovery (`mailbox_recovery.py`), worktrees
(`mailbox_worktrees.py`), watch settings (`mailbox_watch.py`), and the
protected control plane (`mailbox_control_plane.py`). Every cross-file
reference routes through the coordinator's namespace, so each repeated
decision keeps exactly one owner. The child-ticket planning this body
describes staged an extraction that is complete; a future extraction
request needs its own ticket with its own boundary.

### What is already fixed

The daemon has strong tests for dispatch, durable state, candidate recovery,
landing, Red Team returns, push debt, restart behavior, and protected
control-plane work. Important operations fail closed and preserve evidence.

### What is missing

Nothing. The split named in the status above is in the tree, its part files
are listed in the machine contract's trusted tools, and the daemon suite and
standalone workflow reproductions run against the split layout.

<details><summary>Technical record for development tools</summary>

Start with a responsibility that already has a narrow input and output, few
global dependencies, and direct regression coverage. Moving code without a
clear reduction in coupling is NO-GO. Do not introduce a second coordinator,
duplicate durable state, change authority, rename states merely for style, or
mix behavior changes into a mechanical extraction.

For every child ticket, retain focused unit tests and at least one full-path
reproduction across the affected authority boundary. Run the existing mailbox,
candidate recovery, landing, push-debt, and restart suites that can interact
with the extracted code. The old and new paths must produce the same durable
records and refusal behavior before the extraction is accepted.

</details>

<a id="open-github-push-choice"></a>
## Let the user choose whether accepted work is pushed to GitHub

### High-level summary

An accepted ticket currently reaches the local `main` branch and the daemon
then tries to push that exact commit to GitHub. A user may instead want to
inspect or combine locally accepted work before sending anything to the remote
repository.

Add a watcher option `--github yes|no`. Both choices must keep the existing
review and local landing process: the accepted change is still merged into
local `main`. With `yes`, the daemon performs its existing non-force push and
remote verification. With `no`, it stops after the verified local merge and
does not describe that intentional choice as failed push debt.

### Current status

**Ticket type: NEW FUNCTIONALITY.**

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**CLOSED.** The watcher accepts `--github yes|no` with the documented
default `yes`, so existing commands keep pushing unchanged. The choice is
read in exactly one place — inside the push function that every landing
kind and every debt retry already calls — so the local landing path is
byte-identical for both values. With `no`, the function returns before any
Git command: nothing contacts the remote, one sentence names the verified
local landing and the user choice, no push-debt record is written for that
choice, and debt records from earlier runs stay on disk. The three callers
that report push outcomes distinguish the intentional skip from failed-push
debt, and the watch banner states the choice once at startup. The daemon's
self-restart re-executes the original command line, so the value survives a
restart of the active watch. Tests in
`ai/tests/test_role_workflow_behavior.py` prove the skip contacts no
remote against a deliberately missing repository path, preserves an earlier
debt file byte for byte, keeps the default at `yes`, and that an unknown
value fails at command-line parsing before any work; the existing
non-force-push test keeps covering `yes`.

### What is already fixed

The daemon creates and verifies the landing commit, advances local `main`
without force, attempts one exact push, verifies remote `main`, and records
durable follow-up information when an intended push fails.

### What is missing

Nothing. `ai/README.md` explains the choice beside the watch example,
`ai/tools/README.md` explains it beside the push-debt record, and daemon
ownership of L, remote verification under `yes`, and the force-push
prohibition are untouched.

<a id="open-landing-backlog-identity"></a>
## Bind each landing to its candidate and sealed backlog

### High-level summary

An accepted landing intentionally combines two reviewed changes: candidate C
contains the Implementer's fix, while the Architect supplies the sealed
backlog update that closes the ticket. The daemon already verifies both, but
the machine contract still describes the landing mainly as the audited
candidate delta.

Record the complete relationship explicitly so a later audit can answer one
plain question: which candidate and which exact backlog bytes produced this
landing commit?

### Current status

**Ticket type: NEW FUNCTIONALITY.**

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**CLOSED — Git already stores the answer the ticket wants recorded.** The
question "which candidate and which exact backlog bytes produced this
landing commit" is answerable from the repository alone: durable landing
state names candidate C and landing L, and the exact backlog bytes are the
landing commit's own tree, so their digest is recomputable at any time with
`git show L:ai/notes/backlog.md`. The daemon verifies the sealed overlay
before it builds L, which is the moment verification can still refuse.
Writing the digest again into contract fields, commit trailers, and a saved
tuple would create a second record that can only ever agree with the tree
or falsely disagree with it, and every added disagreement case needs its
own recovery handling — bookkeeping surface without a failure it prevents.

### What is already fixed

The daemon accepts only exact candidate C, checks the sealed backlog, builds L
with one parent, and verifies that L contains the audited candidate change plus
the permitted backlog update. The Implementer cannot edit or seal the backlog.

### What is missing

Nothing. The pre-landing overlay verification, the one-parent audited-delta
rules, and the landing tree itself already carry the complete relationship;
a duplicate digest record is declined as a second source of truth.

<a id="open-backlog-sync-crash-cuts"></a>
## Test every interrupted backlog synchronization step

### High-level summary

When the Architect worktree advances, the daemon temporarily preserves the
sealed backlog so Git can update the rest of the checkout without losing the
ticket record. The recovery code is careful, but current tests exercise normal
synchronization rather than stopping the process after each filesystem and Git
step.

Add fault-injection tests that interrupt those exact boundaries and restart
the same routine. Every restart must either recover the one accepted backlog
or stop with a clear conflict; it must never discard or silently choose between
different bytes.

### Current status

**Ticket type: NEW FUNCTIONALITY.**

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**CLOSED — the test cannot be built without breaking a stronger rule.**
Stopping the process exactly between the synchronization boundaries — after
the move to recovery, after the tracked restore, after the fast-forward,
before the recovery-file delete — requires either replacing `os.replace`,
the trusted `git restore`, or `os.unlink` while the routine runs, which is
a monkey patch the Python contract prohibits in tests, or planting
injection hooks inside the production synchronization code, which changes
the trusted path in order to test it. Timing a kill from outside cannot hit
those exact boundaries reliably. The end states the cuts would produce are
already covered by the existing checks: equal bytes converge, conflicting
bytes fail closed and stay preserved, and the guard binds the exact
accepted bytes. With no synchronization failure ever reproduced, the
missing evidence does not justify either repair form.

### What is already fixed

The recovery file is private, the backlog guard binds exact accepted bytes,
and a normal landing preserves both candidate C and the Architect's tracked
backlog update. Conflicting legacy and tracked bytes already fail closed.

### What is missing

Nothing. The fault-injection reproduction is declined because every honest
construction of it either adds a prohibited monkey patch or plants test
hooks in the trusted synchronization path; the recoverable end states keep
their existing coverage.

<a id="open-ai-ticket-latex-guide"></a>
## Write a LaTeX guide to the AI ticket system

### High-level summary

The repository has a long-form LaTeX manuscript that teaches the emulator
library from its inputs through training and scientific checks. The AI ticket
system now needs a companion manuscript. A reader should be able to understand
why the system exists, how one request becomes one tested ticket, and which
role owns each decision without first learning AI-agent or Git terminology.

Create the new source and compiled PDF under `documentation/`. Follow the
teaching quality and visual care of `documentation/emulator_code_guide.tex`,
but explain the ticket system under `ai/` rather than the emulator's
scientific calculations.

### Current status

**Ticket type: NEW FUNCTIONALITY.**

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**CLOSED.** The manuscript exists as `documentation/cocoa_flow_guide.tex`
with its tracked compiled `documentation/cocoa_flow_guide.pdf` (23 two-column
pages). The file is named after the system's own name, CoCoA-Flow, and its
existing frontispiece artwork, rather than the provisional
`ai_ticket_system_guide` name in this ticket. The user explicitly advanced
this ticket ahead of the remaining Open items, which the ordering rule below
permits, and directed the document at teaching a reader how to read the
`ai/tools/` code with the same discipline as the emulator manuscript. The
delivered structure follows that direction: a notation section for the commit
labels C, L, M0/M1, B/P, and D0/D1 with the recurring vocabulary; an
end-to-end ordinary-ticket walkthrough; one section per module family (the
daemon split and its single namespace, the role contract and its compiled
safety floor, the three standalone guards plus candidate admission, message
envelopes, the store's atomic claims and locks, the watch rendezvous and
cycle barriers, the durable cycle record, dispatch and the provider surround,
the handoff-contract grammar, the manual relay, worktree provisioning and the
backlog carry-forward, candidate-to-landing, recovery, the protected
boundary with the two-key ritual and the D0/D1 rehearsal, ticket and
reopening rules, and bundles); verbatim code excerpts and exact refusal
messages throughout; and a staged file-by-file study route with three shorter
role-specific routes. The beginner-operator tutorial sketched below stays
owned by `ai/README.md` and `ai/tools/README.md`, which the manuscript names
as its reading stage 0 and does not duplicate; the sketch's separate worked
figures are replaced by the walkthrough, tables, and code excerpts in the
emulator guide's visual style. Built with `latexmk -pdf
-output-directory=documentation documentation/cocoa_flow_guide.tex` from the
repository top (the frontispiece path is repository-root-relative); the
final log has zero undefined references and zero overfull lines, and every
rendered page was inspected. `documentation/README.md` records the catalog
row and the build-and-inspect command.

**Priority: LOWEST.** This was the final item in the backlog, held behind
every other ticket unless the user explicitly changed the ordering; the user
did so when ordering the manuscript.

### What is already fixed

`ai/README.md`, `ai/tools/README.md`, the role files, and the eleven permanent
notes contain the current behavior and the human-first writing contracts. The
emulator manuscript supplies an established example for LaTeX structure,
figures, captions, appendices, PDF publication, and rendered-page review.

### What is missing

Nothing. The tracked manuscript and PDF exist, the closure paragraph above
records where the delivered form deviates from the original sketch and why,
and `documentation/README.md` carries the catalog row and the exact
build-and-inspect command. Whether an additional beginner-operator tutorial
with worked figures is wanted is a separate future request, not an unmet part
of this ticket.

# Parked edge cases

- PARKED **LOW — EDGE CASE** **BUG FIX** — [Remove hidden covariance files left by forced process termination](#parked-cmb-covariance-cleanup)
- PARKED **LOW — EDGE CASE** **BUG FIX** — [Certify the vendored Syren formulas independently](#parked-syren-formula-certificate)
- PARKED **LOW — EDGE CASE** **BUG FIX** — [Guard the sizing probe if a stateful-forward family is added](#parked-memory-probe-stateful-forward)

<a id="parked-memory-probe-stateful-forward"></a>
## Guard the sizing probe if a stateful-forward family is added

### High-level summary

The batch-memory estimate runs one dummy forward on zeros through the live
model to measure its autograd-saved activations. The current model families
change no state during a forward pass, so the probe is harmless. A future
family containing batch-normalization running statistics or active dropout
would make that probe mutate model state or consume random numbers before
training begins.

### Current status

**Ticket type: BUG FIX.**

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**PARKED. Severity: LOW — EDGE CASE.** No supported design contains such a
module, so this ticket is below Low and is not actionable unless the user
explicitly asks the Architect to solve this ticket by name.

### What is already fixed

Every current design's forward pass is stateless: the probe on zeros with
scoped saved-tensor hooks changes nothing and draws no random numbers.

### What is missing

No automatic work is authorized. If a stateful-forward family is added,
run the probe under an evaluation-mode guard that restores training mode,
or measure on a throwaway copy. Do not build a general state-digest
framework for module types the library does not use.

<details><summary>Technical record for development tools</summary>
Owner: `emulator/batching.py::compute_batch_byte_terms`. The trigger is any
design registering a batch-normalization module or a dropout module with a
nonzero rate.
</details>

<a id="parked-cmb-covariance-cleanup"></a>
## Remove hidden covariance files left by forced process termination

### High-level summary

The covariance writer keeps partial bytes under a hidden private name and
removes that name after a normal success or handled failure. A forced
termination such as `SIGKILL` does not let Python run its cleanup block, so an
unreferenced hidden file can remain in the output directory. Readers never use
that private name, and no partial final covariance is exposed.

### Current status

**Ticket type: BUG FIX.**

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**PARKED. Severity: LOW — EDGE CASE.** This ticket is below Low and is not
actionable unless the user explicitly asks the Architect to solve this ticket
by name.

### What is already fixed

An ordinary write, catchable interruption, rerun, or competing writer cannot
expose a partial final archive or replace a result that already owns the
requested name.

### What is missing

No automatic work is authorized. If the user activates this ticket, add only
a small stale-private-file cleanup rule. Do not restore the former retry,
exact-readback, directory-synchronization, and special exception framework.

<details><summary>Technical record for development tools</summary>

Residual case: `SIGKILL` or another uncatchable termination can leave one
unreferenced hidden temporary file. Activation requires an explicit user
request naming `Remove hidden covariance files left by forced process
termination`.

</details>

<a id="parked-syren-formula-certificate"></a>
## Certify the vendored Syren formulas independently

### High-level summary

Matter-power artifacts record which Syren formula supplies their analytic
starting surface. If a future edit changed the vendored formula without
retraining the matching artifacts, the learned correction could be combined
with a different starting calculation and still produce finite values.

No such drift is demonstrated in the current library. Building a formula
registry or a broad hash framework merely to guard this hypothetical edit
would make the scientific path harder to read, so the case remains below Low.

### Current status

**Ticket type: BUG FIX.**

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**PARKED. Severity: LOW — EDGE CASE.** This ticket is not actionable unless
the user explicitly asks the Architect to solve it by name.

### What is already fixed

Artifacts save the supported Syren law name. Loading rejects unknown laws,
and serving validates the analytic surfaces and their final composition.

### What is missing

No automatic work is authorized. If activated, prefer a small independent
known-answer or version check over a general identity framework.

<details><summary>Technical record for development tools</summary>

Residual case: a future source edit changes a supported Syren formula while
leaving its name and trained correction artifacts unchanged. Activation
requires an explicit request naming `Certify the vendored Syren formulas
independently`.

</details>

# Closed tickets

Closed tickets are grouped by subject. A closed ticket has no missing work of
its own. If later work is still required, **What is missing** links to one of
the open tickets above.

<a id="closed-role-context-separation"></a>
## Give each AI role its own context limit

### High-level summary

The Architect and Implementer previously shared one context setting. A large
Architect allowance could therefore exceed an Ollama Implementer's smaller
model context and make even the connection check fail before work began.

Each role now has an independent command-line setting. Choosing a large
Architect context no longer changes the Implementer threshold.

### Current status

**Ticket type: BUG FIX.**

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**CLOSED. Severity: CRITICAL.** The shared setting prevented the selected
three-role configuration from starting. Ping, runtime validation, saved ticket
identity, dispatch environments, terminal reports, tests, and guides now use
the Implementer's own value.

### What is already fixed

`--architect-context`, `--implementer-context`, and `--sol-context` control
only their named roles. The older `--claude-context` spelling remains as an
Architect-only compatibility name. Ollama still reports the model's actual
maximum separately and refuses an Implementer threshold above that maximum.

### What is missing

Nothing for this ticket.

<details><summary>Technical record for development tools</summary>

Focused tests prove that a 300000-token Architect threshold and a 64000-token
Implementer threshold remain distinct during ping, Ollama preflight, saved
runtime validation, and dispatch selection.

</details>

<a id="closed-control-plane-live-state-compatibility"></a>
## Test a proposed controller against the saved state it must inherit

### High-level summary

A replacement mailbox controller must understand work saved by the controller
that is running now. Testing only a new empty repository could approve a new
file format that fails as soon as it encounters a real active ticket,
completed landing, candidate, or recovery record.

The protected check now begins with a disposable copy of the current records.
The proposed controller must read those copies in a fresh process and preserve
their exact workflow meaning before the separate clean-start scenarios run.

### Current status

**Ticket type: BUG FIX.**

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**CLOSED.** D0 copies ticket and candidate state, saved role-worktree records,
the backlog and recovery records, pending push or note-administration records,
and private candidate or landing refs into an isolated checkout. D1 must read
the copies without changing D0's live bytes or refs.

**Severity: HIGH.** An incompatible protected upgrade could halt the central
mailbox controller after landing and require recovery with the old controller.

### What is already fixed

The D0-owned test compares every active and completed cycle, every candidate
and landing identity, all copied recovery bytes, all saved worktree identities,
and every private ref. A state-schema change requires an exact migration
declaration. D0 runs the named migration only on the copy, starts a fresh D1
process, and rejects missing declarations, wrong versions, or a migration that
drops saved work. The older fresh-state, restart, landing, stale-main, and
health checks still run afterward.

### What is missing

Nothing for this ticket.

<details><summary>Technical record for development tools</summary>

Focused coverage is in
`ai/tests/test_protected_control_plane_shadow.py`. It seeds active and completed
D0 cycles, a candidate ref, push debt, backlog recovery, and three worktree
records; proves the current handoff succeeds; and supplies negative D1 examples
for an undeclared schema, incorrect migration versions, and a migration that
deletes the active ticket. The outer D0 fingerprint covers every copied live
record and the relevant Git refs before and after the disposable run.

</details>

<a id="open-syren-amplitude-aliases"></a>
## Refuse conflicting amplitude names before calculating Syren matter power

### High-level summary

The Syren matter-power formulas accept the primordial amplitude either as
`As`, the usual small number, or as `As_1e9 = 10^9 As`. A Cobaya run may make
both names available because different theory components use different
spellings.

When both names are present, the current helper chooses `As_1e9` without
checking whether it agrees with `As`. A saved network can therefore read one
amplitude while its analytic Syren starting surface uses another. Both results
remain finite, but their combination describes no single cosmology.

This affects the central matter-power prediction rather than a plot or
optional report. A concrete conflicting pair in the permanent scientific note
changes the analytic linear-power baseline by about 77 percent.

### Current status

**Ticket type: BUG FIX.**

**Red Team reopen count: 1.**

**Red Team reopening: barred by Architect NO-GO.**

**Red Team reopen 1 — Architect NO-GO (2026-07-19).** The Red Team correctly
observed that `emul_mps.calculate` calls the linear learned predictor
(`p_lin.predict`) once before `syren_params_from` raises, so the earlier
"before any learned predictor" wording overstated the boundary (corrected
below). The durable acceptance authority (`artifacts-inference-warmstart.md`,
"Syren parameter aliases must agree", rules 2 and 9 and the amplitude
paragraph) requires refusal before either Syren analytic formula and leaves no
`Pk_grid`/interpolator/derived state on failure; the landing meets both — both
`base_pklin`/`base_boost` and the boost predictor never run and `state` stays
empty. The finding is Red-Team Low, below the ticket's user-set High severity,
and causes no wrong science, data loss, or halt; the "unmeasurable"
exact-landing budget is a squash-measurement artifact (the binding pre-landing
candidate budget was `within limit`: added 8625, deleted 20, total 8645, limit
15000). Reopening is permanently barred. Full adjudication in
`ai/notes/ticket-syren-amplitude-aliases.md`.

**CLOSED.** Accepted candidate `f30f406ee826a1a1222282370933e62b9837031b` adds
the both-present amplitude-agreement guard to
`emulator/syren_base.py::syren_params_from`, so a conflicting `As_1e9`/`As`
pair raises a `ValueError` that names both values before either Syren analytic
formula (`base_pklin`/`base_boost`) runs and before any `Pk_grid`,
interpolator, generator row, or Cobaya derived state is written. Single-name
and consistent two-name inputs keep their previous numerics. The complete
directive and audit record are in `ai/notes/ticket-syren-amplitude-aliases.md`.

**Severity: HIGH.** The mismatch can silently change the analytic starting
surface used in a central matter-power result. Medium is insufficient because
the current public EMUL2 configuration makes both amplitude names a normal
input shape, and the numerical error remains finite instead of stopping.

### What is already fixed

`syren_params_from` compares `As_1e9` with `1e9 * As` under one documented
absolute-and-relative tolerance sized to float32 storage and refuses a
disagreeing pair with a `ValueError` that names both supplied values and the
conversion. Because both production call sites invoke this one function before
any Syren formula, the generator refuses a conflicting sample before a raw or
base row is written and the adapter refuses before any `Pk_grid` or derived
state exists. `As`-only and `As_1e9`-only inputs keep their previous numerics.

### What is missing

Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Severity: HIGH BUG FIX. Accepted candidate
`f30f406ee826a1a1222282370933e62b9837031b` on cycle
`open-syren-amplitude-aliases@23d0340587bde038008ddcfa71eea2f2d658ed54`. Owner:
`emulator/syren_base.py::syren_params_from`; witnesses in
`ai/tests/test_syren_dark_energy_coordinates.py`,
`ai/tests/test_generator_dark_energy_facts.py`, and the new
`ai/tests/test_mps_amplitude_aliases.py`. The repair keeps `As`-only and
`As_1e9`-only numerics unchanged, compares `As_1e9` with `1e9 * As` under
module constants `SYREN_AMPLITUDE_ATOL`/`SYREN_AMPLITUDE_RTOL`, names both
supplied values on refusal, and proves no raw/base row or partial `Pk` state
survives a conflict. The two production call sites and the MPS generator and
adapter code are unchanged; the guard lives only in `syren_base.py`.
The audit re-ran every check against the immutable candidate: `py_compile` on
all four changed files; 18, 15, and 2 tests in the three affected modules;
`permanent_note_guard.py` PASS at the base; and the architect-audit
ticket-change guard reporting `within limit` (added 8625, deleted 20, total
8645, limit 15000). The durable rule already existed in
`ai/notes/artifacts-inference-warmstart.md` ("Syren parameter aliases must
agree"), so no permanent note changed.
</details>

<a id="closed-syren-amplitude-aliases"></a>

## Documentation and teaching

### Make backlog tickets and the gate guide readable

**High-level summary.** The backlog previously mixed open and closed work in
one long list, and internal labels such as “unit 8” did not tell a human what
was wrong. The gate guide also compressed tests, gates, board operation, and
workstation rules into language that hid their practical differences.

**Current status.** **CLOSED.** Both guides now pass the human-first contracts
and independent factual review.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The backlog has a linked Open section followed by a
grouped Closed section; every ticket separates its plain summary, status,
completed work, missing work, and technical record. The gate guide gives real
commands and visible results, covers every emulator family, and explains the
runner's actual setup, logs, states, restart behavior, and command options.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Landed and pushed as `b906147`. Evidence: 335/335 full CPU tests; 32/32
permanent-note and role-contract tests; focused CMB test; live board help,
49-gate list, and dataset-publication dry run; board self-test; exact-commit
eleven-note guard; whitespace and prohibited-word checks; backlog inventory
GO; gate fact-audit GO; and permanent-contract review GO.
</details>

### Remove the obsolete README trimming quota

**High-level summary.** A former ticket required a fixed 15 percent reduction
in README words, even when the words were useful. Detailed teaching now lives
in folder guides, so future trimming is based on clarity instead of a quota.

**Current status.** **CLOSED.** The percentage target has been retired.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** README work may make small natural cuts without
removing examples or explanations merely to reach a number.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
User directive retired the target; no separate implementation commit was
required.
</details>

### Explain every developer test

**High-level summary.** The test guide previously listed filenames with short
labels that did not tell a reader what input was used or why a refusal mattered.
It now explains how to run tests and gives a concrete example, action, accepted
result, refusal, and scientific reason for every test module.

**Current status.** **CLOSED.** The guide and its inventory check are on main.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Every immediate Python file in `ai/tests/` is described,
and an automated check rejects a missing, duplicate, or stale inventory row.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Landed and pushed as `a875f3f`, with the later scientific explanation pass in
`9a55c7b`. Evidence includes 333/333 final AI tests, focused inventory checks,
link/fence checks, permanent-note guard, and two final read-only reviews.
</details>

### Make the YAML workflow diagram readable on phones

**High-level summary.** The first `example_yamls` diagram was too wide for a
phone or tablet. It is now a five-step vertical path with short labels and the
same sequence stated in prose.

**Current status.** **CLOSED.** The vertical diagram is published.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Choose, copy, edit, check, and run appear from top
to bottom without overlapping labels.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Landed and pushed as `ac3b3eb`; 243/243 tests, independent phone-layout review,
diff check, and permanent-note check passed.
</details>

### Move detailed appendices out of the main README

**High-level summary.** The main README required a new user to cross many pages
of specialist material before finding the first run. It now keeps a five-step
startup path and sends YAML, data generation, Cobaya, emulator, Syren, and AI
details to their folder guides.

**Current status.** **CLOSED.** The shorter root guide is on main.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The root README changed from 3,717 to 1,122 lines,
and `emulator/CODE_REFERENCE.md` owns the dense code maps.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Landed and pushed as `b0aa890`; 335 links, 67 shell/YAML/Python examples, six
vertical Mermaid sources, 243/243 tests, board checks, and two reviews passed.
</details>

### Make the permanent notes durable and Python style mandatory

**High-level summary.** The permanent notes read like a dated development
diary and Python readability was treated as a preference. The eleven notes now
record neutral current rules, and readable Python is a required GO/NO-GO
condition.

**Current status.** **CLOSED.** The permanent-note and Python contracts apply.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Only the Architect may edit permanent notes, and
`python-changes-go-no-go.md` protects code intended for students and C users.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Evidence included 30/30 focused contract tests, 6/6 backlog-bundle
reproductions, 243/243 full tests, and scientific, workflow, artifact, role,
link, and whole-diff audits.
</details>

### Correct CoCoA setup and project paths in the guides

**High-level summary.** Four guides duplicated CoCoA setup and used an
invented `projects/lsst_y1/cobaya` folder. They now point to the official CoCoA
instructions, place editable YAML directly under `projects/lsst_y1`, and
separate user-copied YAML from generated data exposed by startup links.

**Current status.** **CLOSED.** The verified paths are in all four guides.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The copy and syntax examples use the real project
layout, while nested CMB and grid paths retain their true resolution rules.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Landed as `b87b9f7f1de4e756d55d232edaeab6758bb20516`; 179/179 tests,
250 links, 215 fragments, 29 Bash fences, runtime examples, note guard, and
factual/cold-reader reviews passed.
</details>

### Add a beginner guide for the AI tools

**High-level summary.** Users had to read the large AI guide to discover which
tool to run and whether it changed files. `ai/tools/README.md` now explains all
five programs, daily commands, visible results, stopping, recovery, and bundle
transfer beside the tools themselves.

**Current status.** **CLOSED.** The tool guide is the command reference.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** `ai/README.md` keeps the first-ticket path, while
advanced runtime controls and recovery live in the tool guide.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Landed as `20862070723aa1b3d2e09d3250cfba717edb7a2d`; 179/179 tests,
181 links, 161 fragments, 45 Bash fences, focused tool reproductions, render,
guard, and two reviews passed.
</details>

### Make all AI README workflows vertical

**High-level summary.** Six AI workflow diagrams were hard to follow on narrow
screens. They now read from top to bottom and use visible actions instead of
unexplained internal labels.

**Current status.** **CLOSED.** All six rendered diagrams are taller than wide.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The prose beside each picture states the same
sequence, so no safety rule depends on interpreting the graph.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
All six final PNGs were inspected; 179/179 tests, link/anchor/fence checks,
hard-zero scans, permanent-note guard, and two reviews passed.
</details>

### Add a beginner guide for Cobaya adapters

**High-level summary.** The `cobaya_theory` folder lacked a direct path from a
saved emulator to one checked Cobaya evaluation. Its guide now covers five
adapters, setup versus evaluation, file matching, device behavior, physical
limits, and the current NumPy 1.x boundary.

**Current status.** **CLOSED.** The six-step guide and appendices are complete.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** A user can choose an adapter, manually copy a real
template to `projects/lsst_y1`, run a setup check, and inspect one result.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Evidence includes README SHA-256 `5df8352818043dcd32c29f73c448f502447871251f0d70c09bf9fd851a305dbf`,
179/179 tests, Bash/YAML/copy/runtime/identity/link/render checks, guard, and
three reviews.
</details>

### Add a beginner guide for example YAML files

**High-level summary.** The ten shipped YAML files had no guide for choosing a
starting point or separating a syntax check from scientific validation. The
new guide shows how to choose, copy, edit, check, and run one template, then
uses appendices for special modes and path rules.

**Current status.** **CLOSED.** Every shipped YAML appears in the chooser.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The guide states that editable YAML is copied by
the user and that a successful parser check does not prove files or scientific
settings are correct.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Evidence: 179/179 tests, ten-file census, 45 links, 42 anchors, Bash/YAML/copy
and help checks, diagram render, permanent-note guard, and three reviews.
</details>

### Simplify the first AI workflow picture

**High-level summary.** The first AI diagram introduced ten boxes before a new
reader understood the basic process. It now shows only the user request,
Architect plan, Implementer work and tests, and Architect review.

**Current status.** **CLOSED.** The four-step introduction is published.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Optional Red Team, repair loops, mailbox files, and
worktrees are explained later instead of crowding the first picture.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
The exact 1173x94 render, 31 links, 18 anchors, six Mermaid fences, 179/179
tests, permanent-note guard, and two reviews passed.
</details>

### Add a beginner guide for generating training data

**High-level summary.** The data-generator programs create the scientific
tables used for training and validation, but the folder had no single guide to
their outputs, failure flags, seeds, memory, MPI, resume, or append behavior.
The new guide gives a six-step first path and moves family detail to plain
question-led appendices.

**Current status.** **CLOSED.** The guide covers all five generator commands.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Training and validation commands are separate, and
executable examples check shapes, failure flags, and row overlap.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Focused data tests 64/64 and full tests 179/179 passed, along with five help
routes, a temporary chain-only example, every fence/link check, three reviews,
and the permanent-note guard.
</details>

### Explain why the three-role system saves scarce AI tokens

**High-level summary.** The AI guide used roles without first explaining why
a student would accept the extra structure. It now says that unlimited access
may make the system unnecessary and explains why expensive reasoning can be
reserved for the Architect and optional Red Team while a simpler model writes
and tests code.

**Current status.** **CLOSED.** The cost rationale opens the AI guide.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Tokens and selectable model roles are defined with
availability-qualified Opus, Sonnet, and Haiku examples.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Links, fences, prose scans, 179/179 tests, permanent-note guard, factual review,
and cold-reader review passed.
</details>

### Rewrite AI appendices in beginner language

**High-level summary.** The AI appendices introduced words such as lane,
dispatch, worktree, and schema before a reader could connect them to a file or
action. FAQs A-H now define or replace those terms where they first appear and
explain stopping, role folders, Red Team scope, recovery, and archive transfer
with concrete examples.

**Current status.** **CLOSED.** The appendix language pass is complete.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Exact watcher messages, positive and zero cycle
behavior, role selection, and bundle hash checks now have local explanations.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Exact-output 8/8, two-role runtime 18/18 with 21 mutations killed, 179/179 full
tests, Mermaid/anchor checks, and two cold-reader reviews passed.
</details>

### Protect README and Python explanations with a GO/NO-GO contract

**High-level summary.** README prose and explanatory Python text could become
factually polished but unusable for a physics student. The eleventh permanent
note now requires concrete examples, local definitions, exact evidence, and
anti-AI writing checks, while the SHA guard prevents accidental note drift.

**Current status.** **CLOSED.** The contract and guard apply to every role.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Implementer and Red Team cannot edit the eleven
notes; the Architect pins and rechecks their exact bytes.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Focused guard/role/handoff tests 41/41, bundle regression 6/6, full tests
179/179, whitespace check, and independent guard review passed.
</details>

### Remove the overlapping mailbox-diagram label

**High-level summary.** A self-loop label in the mailbox lifecycle picture
rendered on top of another label. The loop was removed and the unchanged
failure meaning was written beside the work-in-progress box.

**Current status.** **CLOSED.** The diagram is readable.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The layout and prose now communicate the same
archive behavior without overlapping text.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
README contract and whitespace checks passed.
</details>

### Rename `texnotes` to `documentation`

**High-level summary.** Teaching sources and the activation-function notebook
were stored under an unclear folder name. They now live under
`documentation/`, and links, build paths, and custody text use that name.

**Current status.** **CLOSED.** The rename and rebuilt PDF are complete.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** `activation_functions_teaching.nb` moved with the
documentation tree and the generated guide builds from current paths.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Independent path and build audit issued GO.
</details>

### Make the AI guide role-first and visual

**High-level summary.** The AI guide mixed role rules, command details, and
internal mechanics before explaining who decides and who changes code. It now
starts with stable role boundaries, selectable Claude models, Architect-only
GO/NO-GO, bounded Red Team scope, and visual workflows.

**Current status.** **CLOSED.** The role-first guide is on main.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Exact tool reference moved to the tools guide, and
shorter paragraphs plus diagrams explain runtime behavior.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Output parity 8/8, fix-only 14/14 with 20 mutations, role models 5/5 with five
mutations, safe stop 9/9 with seven mutations, help/anchor/fence/SVG and diff
checks passed.
</details>

### Render README equations correctly

**High-level summary.** Four formulas appeared as raw bracketed text instead
of rendered mathematics. Both delta-chi-square equations and the default
activation equation now use GitHub-compatible display-math fences.

**Current status.** **CLOSED.** All four formulas render.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Only the Markdown delimiters changed; equation
bodies and scientific prose stayed the same.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Landed and pushed as `66f7046`; 130 links/assets, math and Markdown fence
checks, permanent-note guard, and whitespace check passed.
</details>

### Reorganize the root and AI READMEs around short startup paths

**High-level summary.** Both large guides mixed first-use instructions with
reference material. They now separate a short main path from question-led
appendices grouped by subject.

**Current status.** **CLOSED.** The reader-path rewrite is on main.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The later `c91791a` hygiene change also repaired
the disposable primary-worktree test fixture that this ticket originally left
behind.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Landed and pushed as `eb17489`; 110 links, 55 anchors, 42 YAML fences, mailbox
reproductions, 84/84 tests, diff check, and three reviews passed. The old
scratch-fixture debt is superseded by `c91791a`.
</details>

<a id="closed-ollama-documentation-model"></a>
## Use GLM-5.2 Cloud in the Ollama examples

### High-level summary

The AI guides used Qwen as their example Ollama Implementer. The preferred
documented choice is now `glm-5.2:cloud`, so a user copying either the watcher
or connection-check command should see that model consistently.

### Current status

**CLOSED.** This was accepted as a **LOW DOCUMENTATION CHANGE**. It changes
the recommended examples, not the daemon's Claude default or the user's
ability to name another Ollama model.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

### What is already fixed

The short AI guide and detailed tool guide use `glm-5.2:cloud` for Ollama
watch and `--ping` commands. They also explain that this choice requires an
Ollama account and processes prompts through Ollama's cloud service.

### What is missing

Nothing for this ticket.

<details><summary>Technical record for development tools</summary>

Only documentation examples changed. Runtime model selection remains explicit
through `--implementer-provider ollama --implementer-model MODEL`; no model
name was compiled into the daemon.

</details>

<a id="closed-ollama-ping-visible-thinking"></a>
## Let reasoning-capable Ollama models pass the connection check

### High-level summary

The Ollama connection check required the model's entire visible answer to be
the requested marker. GLM-5.2 Cloud answered correctly, but it printed its
reasoning first, so CocoaFlow reported a healthy signed-in model as
unavailable.

### Current status

**CLOSED.** This was accepted as a **HIGH BUG FIX** because the documented
Ollama model could not pass the check required before an unattended watch.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

### What is already fixed

The probe now asks Ollama to hide model reasoning. It still requires an exact,
unpredictable reply, so echoed prompts and unrelated output remain failures.

### What is missing

Nothing for this ticket.

<details><summary>Technical record for development tools</summary>

The repair adds Ollama's `--hidethinking` option only to the no-work
connection probe. The focused provider test requires that option while the
existing nonce and echoed-prompt tests retain the strict response boundary.

</details>

## AI roles, user controls, and handoffs

### Recover a completed Implementer return after validation refuses it

**High-level summary.** The Implementer completed candidate C, but harmless
Markdown list formatting made the evidence validator reject the return. The
watcher then stayed idle instead of preserving the completed work.

**Current status.** **CLOSED. HIGH BUG FIX.** A restart now revalidates the
saved return and sends it to the Architect without rerunning the Implementer.
The evidence parser accepts the same required heading inside a Markdown list.

**Red Team reopen count: 0. Red Team reopening: allowed.**

**What is already fixed.** Candidate identity, planned subagent names, fields,
scope, and clean Git state remain exact. The character guard can also measure
the existing 8,645-character candidate within the selected 15,000 limit.

**What is missing.** Nothing for this ticket.

### Restart cheap role work without rebuilding the Architect plan

**High-level summary.** Ctrl-C can leave an Implementer or Red Team request
outside the waiting queue with unfinished work. Recovery should discard that
role's partial attempt and reuse the Architect's saved plan.

**Current status.** **CLOSED.** This was accepted as a **HIGH BUG FIX** because
the old manual recovery could halt normal ticket operation and invited unsafe
file moves.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** `--restart-implementer` and `--restart-redteam`
requeue one exact handoff. They refuse ambiguity or a completed result.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Tests cover partial edits, exact requeueing, completed-result refusal, and the
Architect's sealed backlog.
</details>

### Explain every Architect candidate review in the terminal

**High-level summary.** A formal `GO` or `NO-GO` says whether work advances,
but it does not tell a human how close the Implementer came. The Architect now
ends each candidate audit with a short assessment of the exact result. The
assessment names strengths, remaining work, file scope, and the next action.

**Current status.** **CLOSED.** This was accepted as **MEDIUM NEW
FUNCTIONALITY** requested for immediate use.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Five plain result categories avoid false numerical
precision. The seven-line assessment fits inside the terminal's existing
eight-line relay tail and leaves the secure decision-only GO message intact.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
The role-contract test pins the ordered terminal rows and the rule that the
assessment judges candidate C rather than the selected model.
</details>

<a id="closed-subagent-discretion"></a>
### Let the Architect decide when helpers add real value

**High-level summary.** Requiring a helper for every ticket could spend more
model credits coordinating a five-line correction than performing it. The
Architect now decides whether another AI session can produce an independent
result, while the Implementer remains unable to skip required work.

**Current status.** **CLOSED.** Every directive contains exactly one visible
choice: `Subagents required` with bounded named jobs, or `Subagents not
required` with a concrete Architect-authored reason.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The handoff validator refuses empty, vague,
contradictory, or Implementer-authored waivers. A no-helper handoff must repeat
the saved reason exactly. Required helpers still launch before the
Implementer's own edit, and the existing SHA-bound capability exception remains
available only after a real required launch fails. The router and daemon
witnesses cover both exact acceptance and changed-reason refusal.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>

Landed and pushed as `0ff77fa`. The one required adversarial review found one
extra closing bracket in the Implementer template. The Architect corrected it,
added an exact regression assertion, reran 94 focused tests, and made the final
decision without a second review round. The complete configured CoCoA suite
passed 790 tests; the router and daemon scratch reproductions also passed.

</details>

<a id="closed-structured-role-contract"></a>
### Protect one machine-readable role contract

**High-level summary.** Stable role permissions, timing limits, candidate
identity, landing authority, backlog ownership, and the single-review rule had
been repeated across Python and long prose. A later edit could change one copy
without making the contradiction obvious to the Architect.

**Current status.** **CLOSED.** The protected JSON-compatible YAML file is now
the small machine source of truth. Live watcher controls must agree with it,
and an Implementer candidate cannot change the contract, its reader, the
eleven permanent notes, their guard, or protected role files.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The strict reader refuses duplicate, missing,
unknown, wrongly typed, noncanonical, oversized, linked, or non-regular input.
The permanent-note guard protects the YAML without changing the eleven-note
Markdown census. A protected-policy draft receives one adversarial GO/NO-GO
recommendation; the Architect decides, and a correction gets no second round.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>

Landed and pushed as `8611e1e`. The sole adversarial review verified staged
diff `df8be80b1ec26a9f146c1b835391ea33f218faac6ff4e529570de56d879231e7`
and returned GO. All 795 configured CoCoA tests, the full disposable-worktree
mutation reproduction, and all six backlog-bundle scenarios passed.

</details>

### Prohibit new monkey patches without forcing a wholesale rewrite

**High-level summary.** A monkey patch changes existing executable behavior
while Python is running, so an apparently local test can change a later test.
The permanent Python contract and all three role contracts now reject a monkey
patch that is added, copied, retargeted, or broadened.

**Current status.** **CLOSED.** Existing sites are not one repository-wide
Critical rewrite. When bounded work encounters one, the Architect records one
separate High bug ticket and keeps the current ticket narrow.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The contract defines prohibited runtime
replacement, names ordinary local fakes that remain allowed, and prevents the
Implementer or Red Team from proposing patching as a shortcut.

**What is missing.** Nothing for the future-rule ticket. Two existing sites
encountered while writing the rule are recorded in the High queue.

<details><summary>Technical record for development tools</summary>
The role-contract test requires the rule in the Architect, Implementer, Red
Team, and permanent Python surfaces. The scoped review rejected a baseline
census and wholesale migration as disproportionate.
</details>

### Use a 4,000-character warning for one bug repair

**High-level summary.** The earlier production-size warning was close to the
size of only a few clear Python lines and made ordinary bounded repairs look
disproportionate too early. The Architect now becomes strongly suspicious
only above 4,000 added-plus-deleted characters outside tests and gates.

**Current status.** **CLOSED.** The number remains a warning, not a hard size
limit. Readability, directness, and the separate `--max` value still govern
every candidate.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The permanent Python contract, Architect role,
entry command, and regression test use the same 4,000-character value.

**What is missing.** Nothing for this policy adjustment.

<details><summary>Technical record for development tools</summary>
The calculation excludes all of `ai/tests/` and `ai/gates/`, which are
evidence surfaces rather than mature scientific production code.
</details>

### Make the Architect the only user-facing role

**High-level summary.** Direct messages to Implementer or Red Team bypassed
the role that owns scope and final decisions. Public send and ping commands now
accept only Architect requests, and the other roles return their work through
the Architect.

**Current status.** **CLOSED.** The user-to-Architect boundary is enforced.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Widespread Red Team work begins only when the user
asks the Architect to request it, and a carried handoff must remain unchanged.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Six mismatch cases refuse with zero writes; 234/234 tests and the role, router,
permanent-note, compilation, diff, and README checks passed.
</details>

### Filter Red Team discoveries by severity

**High-level summary.** Discovery runs could create tickets for edge cases the
user did not want to pursue. The `--severity` setting now lets the user request
only severe failures, probable normal-use bugs, or every concrete finding.

**Current status.** **CLOSED.** The default is `medium`, and the Architect
makes the final severity decision.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Red Team records severity, likelihood, evidence,
and whether the finding meets the user's threshold; Architect may accept,
upgrade, or downgrade it with a reason.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Evidence: 223 tests plus severity, manual-router, worktree, fix-only, two-role,
and character-limit reproductions; permanent-note and README contracts passed.
</details>

### Limit changed characters without allowing unreadable code

**High-level summary.** Maintenance tickets needed a way to reject changes
that touch too much code. `--max` now limits added plus deleted characters, but
the Architect must still reject shortened names, collapsed logic, or removed
explanations that make Python difficult to read.

**Current status.** **CLOSED.** The limit reaches every role and final review.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Zero means unlimited, positive values use the exact
ticket base, and an unmeasurable or conflicting candidate refuses.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Landed as `859dab2`; focused contract/guard/role tests 67/67, full tests
213/213, max propagation 9/9 with four mutations, topology/no-red-team/cycle
and all preserved router suites passed.
</details>

### Give Sol its own saved worktree

**High-level summary.** Sol formerly started in the repository folder reserved
for the user. It now creates and reuses an independent `mailbox-sol` worktree,
so ordinary agent work cannot change the user's main folder.

**Current status.** **CLOSED.** Sol and Claude have separate saved work folders.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Repository, path, branch, role, tool, and notes
identity are checked before and after each Sol launch.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Landed as `1e17fe2`; topology/race runtime 19/19 with mutations killed, all
related daemon suites, security review, and documentation review passed.
</details>

### Create and reuse Claude's primary coordination worktree

**High-level summary.** Claude sessions needed a persistent work folder instead
of guessing or using the user's main checkout. The first live use now creates
or deliberately adopts one worktree and later runs validate and reuse its exact
repository, path, and branch.

**Current status.** **CLOSED.** Creation, reuse, refusal, and migration rules are
implemented.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Concurrent launchers converge, invalid folders fail
without fallback, and archived transport can be copied only under bounded
byte checks. The old disposable test-fixture drift was repaired by `c91791a`.

**What is missing.** No code is missing. A user who owns the preserved
`amazing-keller-e798b6` transport must deliberately migrate or adopt it; the
tool will not mutate that user-owned state automatically.

<details><summary>Technical record for development tools</summary>
Initial evidence: 15/15 focused runtime arms with source mutations, preserved
router suites, 44/44 tests, board self-test, compilation, and diff check. The
real pre-migration `amazing-keller-e798b6` transport remains preserved by
design and requires deliberate user migration rather than automatic mutation.
</details>

### Package unfinished backlog work for another developer

**High-level summary.** A user who runs out of credits may need to send one
snapshot of unfinished work to another developer. `backlog_bundle.py` now
creates a `.tar.xz` package, checks it without writing, and imports it only
into a new ignored review folder.

**Current status.** **CLOSED.** Deterministic package, read, and import modes are
available.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The archive binds repository and base identity,
sizes, SHA-256 values, paths, and exact bytes, and refuses overwrite, links,
special files, traversal, races, malformed manifests, and extra XZ streams.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Focused regression 6/6 on Python 3.14 and 3.9, 44/44 tests, mailbox checks,
compilation, diff, and independent security/policy reviews passed.
</details>

### Allow a two-role run without Red Team

**High-level summary.** Some work needs only an Architect and Implementer, but
the watcher always assumed a Red Team route. `--skip-redteam` and
`--no-red-team` now disable that route while preserving its waiting messages
for a later three-role run.

**Current status.** **CLOSED.** The two-role topology is supported.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Evidence returns directly to the Architect, Sol
sends refuse during the two-role watch, and demand plus cycle-zero count only
enabled routes.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Focused runtime 18/18 with 21 mutations on system, CoCoA, and macOS Python;
manual-router parity, preserved daemon suites, 44/44 tests, board, compilation,
render, and diff checks passed.
</details>

### Let the user choose Claude models by role

**High-level summary.** Architect and Implementer were tied to expensive
default model names. `--architect-model` and `--implementer-model` now accept
aliases or full Claude model IDs while the mailbox routes keep their stable
role meaning.

**Current status.** **CLOSED.** Architect and Implementer models are selected
independently.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The expected Opus-Architect and Sonnet-Implementer
pair is supported, invalid names refuse before launch, and Sol remains
unchanged.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Focused runtime 5/5, five mutations killed, and every preserved daemon suite
passed.
</details>

### Require detailed plans that simpler Implementers can follow

**High-level summary.** Earlier handoffs assumed that an expensive Implementer
would fill in missing design choices. Architect and Red Team directives must
now name exact files, algorithms, invariants, failures, tests, expected
results, exclusions, and stop conditions so a simpler model can execute them.

**Current status.** **CLOSED.** Incomplete or choice-leaving packets refuse.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Detailed deterministic alternatives remain allowed,
but Red Team repair proposals return to the Architect and never execute by
themselves.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Landed and pushed as `866b30b`; focused role/contract/preamble tests 31/31,
full tests 169/169, eleven router reproductions, 59-case security matrix,
compilation, diff, and three reviews passed.
</details>

<a id="closed-failure-catalog-consistency"></a>
## Keep the Implementer failure catalog synchronized with its controls

### High-level summary

The short Implementer failure catalog correctly points to existing recovery
behavior, but one explanation repeated the current 90-minute setting and its
code references could silently become stale after a function rename.

The catalog now names the configurable role-contract setting instead of its
present value. A documentation check verifies that every catalog identifier is
unique and every named Python file and symbol still exists.

### Current status

**CLOSED.** This was accepted as a **LOW BUG FIX**. Stale catalog text could
mislead a future maintainer, but it never changed runtime behavior or granted
the Implementer additional authority.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

### What is already fixed

The `timed_complexity` entry points to
`role-contract.yaml::runtime.implementer_review_minutes`. The existing
role-contract tests parse the reference catalog as documentation, confirm
unique IDs, and use Python's syntax tree to find each referenced function or
class in its named source file.

### What is missing

Nothing for this ticket.

<details><summary>Technical record for development tools</summary>

The check deliberately does not execute the catalog or derive workflow policy
from it. Code, the validated handoff contract, and `role-contract.yaml` remain
authoritative. The test only prevents broken documentation links and a copied
configuration value from misleading a later reader.

</details>

## Scientific code, data handling, and gates

<a id="open-mps-serving-domain"></a>
<a id="closed-mps-serving-domain"></a>
### Reject matter-power requests outside calibrated ranges

**High-level summary.** Matter-power serving now accepts only finite, ordered
saved axes and finite matching surfaces. A redshift outside the saved interval
always stops. A requested wavenumber tail is allowed only through the existing
logarithmic boundary continuation, controlled by a visible option that is on
by default.

The option can be turned off when the user wants every wavenumber confined to
the saved grid. Values inside the grid use the same interpolation path in
either mode.

**Current status.** **CLOSED.** Commit `a0633ad` contains the bounded repair for
the demonstrated serving-domain failures without adding a calibration
registry or a new scientific framework.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Saved and direct interpolation axes must be
one-dimensional, finite, strictly increasing, and long enough for the spline;
wavenumbers must also be positive. Surfaces must have the exact matching shape
and finite values. Empty, nonfinite, nonpositive, or even slightly
out-of-range queries stop. Explicit log-log wavenumber tails reproduce a
known power law, while disabling the option refuses those tails. The adapter
already validated the assembled linear and nonlinear spectra before
publishing them, so that responsibility did not need another implementation.

**What is missing.** Earlier configuration errors are tracked separately in
[Validate matter-power requests before a run starts](#open-mps-request-contract).
The hypothetical case of a future Syren source edit that keeps the old law
name is parked under [Certify the vendored Syren formulas independently](#parked-syren-formula-certificate).

<details><summary>Technical record for development tools</summary>

Evidence: 811/811 developer tests; 12 focused boundary tests; both adapter
gate groups, with 11/11 strict checks and 23/23 publication checks; the
18-check matter-power adapter assembly leg; Python compilation; whitespace
check; and one required adversarial review. That review found tolerant range
comparisons and inward extrapolation bounds. The final candidate removes both
and adds direct regression witnesses. Production and user-guide changes total
3,949 added-plus-deleted characters; tests and gates are excluded from the
4,000-character warning.

</details>

<a id="closed-background-protocol"></a>
### Reject invalid redshift grids, coordinate pairs, and nonflat cosmologies

**High-level summary.** Background distances now start from an ordered Hubble
grid whose first redshift is zero. The Cobaya bridge keeps each two-redshift
request as a visible pair, and the current flat-only calculation refuses a
background dataset, saved emulator, or directly named Cobaya curvature that
is nonflat. These checks prevent plausible-looking distances from being
served for a calculation this implementation does not support.

**Current status.** **CLOSED.** Commit `a3b345e` contains the bounded repair.
Curved-distance formulas and renamed or transformed curvature parameters
remain user responsibility unless separately requested.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Training and rebuilding reject nonfinite,
negative, duplicate, reversed, or unanchored Hubble grids. The distance
integrator independently checks its grid and Hubble values. Background
generation and serving reject a directly sampled or fixed nonzero `omk`, and
two-redshift requirements and getters require exact ordered `(N, 2)` rows.
The user guide now shows a zero-anchored low-redshift grid.

**What is missing.** Nothing for this ticket. Supporting arbitrary curved
cosmologies or discovering curvature hidden behind a renamed Cobaya
parameter would require a separate feature rather than more defensive
framework code here.

<details><summary>Technical record for development tools</summary>

The exact staged candidate has SHA-256
`5979f4099f1940a824f14c96a28bcd85dd96b568a9d7b7bc1a1db6a2088dc930`.
The one required adversarial review returned NO-GO because two gate fixtures
still began above zero. Those fixtures were corrected, the authenticated
dataset smoke gate was updated to its current published paths, and the
Architect self-audited the resulting candidate without requesting another
review round. The 5,045 changed characters outside tests and gates exceed the
4,000-character warning, but they are small direct checks at the generator,
saved-grid, integrator, artifact-load, and Cobaya-request boundaries rather
than a new framework. All 802 tests and the `bsn-identity`, `bsn-smoke`, and
`transfer-identity` gates pass.

</details>

<a id="closed-cmb-serving-contract"></a>
### Reject physically impossible CMB spectra before serving them

**High-level summary.** The CMB bridge now checks the complete local result
before it gives that result to Cobaya. TT, EE, and PP must be finite and
nonnegative; TE remains signed but cannot exceed the covariance bound where
TT, TE, and EE share a stored multipole. An invalid prediction leaves the
caller's state unchanged instead of publishing a partial result.

**Current status.** **CLOSED.** Commit `2016c40` contains the corrected repair.
Unit conversion and multipole-factor support remain a separate Medium feature.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Every loaded spectrum is checked for NaN and
infinity. Negative TT, EE, or PP values refuse. The covariance comparison
uses only multipoles stored by all three relevant artifacts and remains stable
for very small and very large finite values. One rounding step in the source
prediction's floating representation admits an honest positive-semidefinite
boundary without allowing an arbitrary tolerance.

**What is missing.** Nothing for this ticket. The adapter still deliberately
refuses unit and multipole-factor conversions until their separate ticket is
implemented.

<details><summary>Technical record for development tools</summary>

The corrected staged candidate has SHA-256
`c6e70e64a6cd130d678e27ef3aa65446f1ea21bfc892ffdc74d3d6d732584b42`
and 2,447 changed characters outside tests and gates. The one required
adversarial review returned NO-GO because squaring could underflow or overflow
and because the initial witnesses did not prove the rounding step or common-
multipole rule. The bounded correction compares `abs(TE)` with
`sqrt(TT) * sqrt(EE)` and adds all three witnesses. The Architect self-audited
the corrected candidate without a second review. All 807 tests, 18 focused
tests, and both `adapter-contracts` gate groups pass; changed Python compiles
and `git diff --check` passes.

</details>

<a id="closed-compatibility-manifest-removal"></a>
### Remove the duplicate compatibility manifest from saved emulators

**High-level summary.** Each saved emulator already records the complete model
recipe, geometry state, analytic law, and composition mode used to rebuild it.
A second compatibility manifest copied those facts and added labels such as
`model:...:v1`. Those fixed labels did not inspect or hash the Python
implementation, so they made saving and loading harder to follow without
providing independent protection.

**Current status.** **CLOSED.** New plain and transfer artifacts no longer
write or require the duplicate manifest. The direct rebuilding records and
strict weight loading remain.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The semantic-label registries, manifest builders,
root and embedded-transfer datasets, rebuild checks, and output-identity fields
were removed as one coherent change. The production change deletes 431 lines
and adds 10; splitting the writer, reader, and filename identity would have
left a partly removed file format. This deliberately breaks the brief alpha
format: an artifact written while the manifest existed may need regeneration.
No dual reader or replacement registry was added.

**What is missing.** Nothing for this ticket. The smaller model-recipe reader,
transfer-state digest cleanup, and training-history load cleanup remain
separate Critical tickets above.

<details><summary>Technical record for development tools</summary>

Landed as `8030857`. The one required adversarial review returned GO and
confirmed that model
recipes, geometry and composition facts, analytic laws, transfer checks,
training histories, pair binding, and strict checkpoint loading remain. The
CPU evidence passed 216 distinct unit tests; the transfer, output-identity,
artifact-composition, and fine-tune gates also passed. The plain-and-transfer
witness saves both artifact forms, confirms neither contains the removed
dataset, and rebuilds both. Changed Python compiled and `git diff --check`
passed.

</details>

<a id="closed-model-recipe-simplification"></a>
### Keep only the model recipe checks needed to rebuild an emulator

**High-level summary.** A saved model recipe prevents a later software default
from silently rebuilding a different neural network. The former implementation
also repeated numerical rules already checked by the real constructors and
factories, which made every saved-emulator load harder to understand.

**Current status.** **CLOSED.** The recipe reader now checks the complete,
closed rebuilding description before importing model code. Numerical rules
remain with the constructor or factory that actually uses each value.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Commit `346e65b` renames the module to the clearer
`model_recipe.py`, removes the duplicate constructor-signature registry and
constructor-owned numerical checks, and keeps the six supported classes,
exact saved fields, known factory names, compile choices, positive network
dimensions, class identity, and geometry choice. Saving still requires the
live model's own recipe to equal the claimed recipe. Reopening still checks
geometry widths and loads every learned tensor strictly.

**What is missing.** Nothing for this ticket. Embedded transfer-state hashes
and training-history checks were handled by the following separate Critical
simplifications.

<details><summary>Technical record for development tools</summary>

The exact staged diff received the one required adversarial review and GO.
The focused recipe and preflight suite passed 41 tests. The broader CPU suite
passed 174 tests; compile-recipe CPU controls passed 12/12; transfer identity,
output identity, composition, and fine-tune gates passed 57, 50, 45, and 19
checks respectively. CUDA compilation was unavailable on this workstation.
Changed Python compiled, `git diff --check` passed, the permanent-note guard
passed at the exact commit, and `main` matched `origin/main` after the push.

</details>

<a id="closed-transfer-state-digest-simplification"></a>
### Remove duplicate hashes for embedded transfer-model weights

**High-level summary.** A transfer emulator already stores its base-model
tensors inside the HDF5 file and loads them strictly into the registered model.
The former path also hashed those same tensors several times and copied the
hashes into HDF5 attributes and nested configuration records.

**Current status.** **CLOSED.** Embedded transfer tensors now have one direct
rebuild path. Missing, extra, or wrong-shaped tensors fail strict model loading.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Commit `64fa00a` removes the digest declarations,
structure walks, repeated live/HDF5 comparisons, and tensor-digest helper. It
keeps `state`, refined-only `drifted_state`, the explicit refinement choice,
model recipes, geometry reconstruction, strict tensor loading, direct
`.emul`/`.h5` pair authentication, source artifact and checkpoint identity,
and output identity. A same-shaped value edit inside the HDF5 state is now
explicitly user responsibility.

**What is missing.** Nothing for this ticket. Training-history checks on the
serving path were removed by the following separate Critical simplification.

<details><summary>Technical record for development tools</summary>

The 14,659-character warning override was accepted because the production
change adds 50 characters and deletes 13,072 as one writer/reader/metadata
format cleanup; splitting it would leave an inconsistent saved format. The
single adversarial review returned GO. The full AI test suite passed 795 tests,
the focused artifact set passed 94, and the transfer, composition, and output-
identity gates passed 57, 45, and 50 checks. Documentation style passed 16
tests, changed Python compiled, the permanent-note guard passed at the exact
commit, and `main` matched `origin/main` after the push.

</details>

<a id="closed-training-history-load-simplification"></a>
### Stop revalidating training history while loading an emulator

**High-level summary.** Training histories explain how a completed run
progressed, but they do not define the neural network used for prediction. The
former rebuild path nevertheless interpreted a large optimizer, schedule,
pass-order, and history grammar before it could load a saved model.

**Current status.** **CLOSED.** Reopening now uses only the records needed for
prediction. The training description and curves remain saved provenance.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Commit `a4f8fa8` removes the duplicate training-
policy parser from `results.py`. Before publication, the writer still requires
five finite history arrays with compatible shapes. Reopening does not read the
history group or interpret pass grammar. It keeps the resolved training mapping
as opaque provenance bound into output identity, while preserving model-recipe,
geometry, composition, scientific-record, artifact-pair, and strict-weight
checks.

**What is missing.** Nothing for this ticket. Training-pass construction and
its focused tests remain because they document what work ran; they do not
control reconstruction.

<details><summary>Technical record for development tools</summary>

The one required adversarial review found the code and protected-note change
sound but returned NO-GO for one stale sentence in the public test inventory.
The Architect corrected that sentence and made the final decision without a
second review round. The final 789-test AI suite passed. Seventy-eight focused
artifact tests, the output-identity, composition, transfer, and fine-tune gates,
documentation checks, Python compilation, and whitespace checks passed. CUDA
compilation was unavailable; its 12 CPU controls passed. The size override was
accepted because the change deletes one coupled framework: 126 lines were added
and 808 were removed across code, tests, and documentation.

</details>

<a id="closed-dark-energy-coordinates"></a>
### Preserve time-varying dark energy from data generation through serving

**High-level summary.** Matter-power calculations can describe dark energy
with a present-day value and a second value that says how it changes with
cosmic time. Cobaya may sample their sum, `w0pwa = w + wa`, and calculate
`wa` before a theory component runs.

The old path could overlook that calculated value and silently replace it
with `wa = 0`. This produced a smooth finite spectrum for constant dark
energy even when the user requested a time-varying cosmology.

Generation and serving now share one checked conversion. The saved dataset
states the physical law, every generated Syren starting surface reuses it,
and the Cobaya adapter reconstructs the saved coordinate names from the
physical values before prediction.

**Current status.** **CLOSED.** Commit `32328be` implements and tests the
repair. Architect-only commit `8b7f991` records the exact rule in the
permanent scientific notes.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Direct `w`/`wa` and transformed `w`/`w0pwa`
coordinates resolve to the same pair. Repeated forms must agree under one
documented tolerance, and incomplete inputs receive zero evolution only from
an explicit constant-`w` or cosmological-constant law. Generator facts use
Cobaya's sampled and calculated parameter information; sampled values cannot
borrow theory defaults and appear fixed. The matter-power adapter requests
physical coordinates rather than dropped `w0pwa`, then rebuilds all names
before either learned or analytic prediction. The shipped EMUL2 point now
uses `w = -0.9`, `w0pwa = -0.7`, and therefore `wa = 0.2`.

**What is missing.** Nothing for this ticket. Conflicting amplitude spellings
are a separate High ticket because they involve `As` and `As_1e9`, not the
dark-energy coordinates repaired here.

<details><summary>Technical record for development tools</summary>
The full `ai/tests` discovery ran 796 tests and ended in `OK` with three
skips. Forty-two focused dark-energy tests passed. A real Cobaya 3.5.7 test
proved the nonzero calculated-`wa` route, and the NumPy-1 fixed-facts schema
gate printed `PASS` for every check. Changed Python compiled, the test README
inventory passed, `git diff --check` passed, the implementation audit returned
GO, and the eleven-note guard passed before and after the separate Architect
note update.
</details>

<a id="closed-artifact-recipe-totality"></a>
### Save every model-building setting needed to rebuild a trained model

**High-level summary.** A saved emulator must rebuild the exact model that was
trained. It may not guess from current Python defaults or accept a plausible
recipe that describes different activation curves, layer counts, geometry,
training phases, analytic formulas, or transfer-base weights.

The writer now compares the saved description with the live model and records
the complete executed training plan and transfer state. The reader validates
the model recipe and direct scientific records before it imports or constructs
model components. The training plan remains provenance and is not interpreted
while rebuilding the prediction model.

**Current status.** **CLOSED.** Commit `dd44234` implements the complete
write-and-read contract.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The closed recipe covers all six supported model
classes, their activations, normalization, and constructor values. Parameter
and output geometries, composition, intrinsic-alignment coefficients, and the
analytic target law are saved directly. The writer checks that saved history
arrays are finite and have compatible shapes. Frozen and refined transfer
states use strict model loading, and schema 3 publication refuses any rescaling
mode that public inference cannot reverse.

**What is missing.** Nothing for this ticket. CUDA compilation of saved modes
still belongs to the separate workstation ticket because this host has no
CUDA device.

<details><summary>Technical record for development tools</summary>
Evidence: 753 developer tests passed with 2 skips; 169 focused artifact tests;
all transfer-identity legs; validation-board self-test; CPU compile-recipe
controls; Python compilation; whitespace check; exact-commit permanent-note
guard; and two independent GO reviews. The CUDA compile lane returned its
explicit unavailable status instead of a false pass.
</details>

<a id="closed-generator-ingress"></a>
### Validate generator inputs before creating output files

**High-level summary.** A malformed parameter order, covariance, fiducial
value, grid, prior bound, or command setting could previously be discovered
after output work had begun. MCMC rows could also appear distinct in memory
but collapse to duplicates when saved as `float32`, leaving a smaller usable
dataset than the user requested.

The generator now validates the complete request before it creates output.
After sampling, it counts distinct rows at the precision readers actually
receive and refuses before creating a draft when that count is too small.

**Current status.** **CLOSED.** Commit `9d53a51` validates fresh input before
publication and validates resume or append state read-only before any new
locator or draft is created.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Parameter names, covariance paths and matrices,
fiducials, prior-bound conversion, shared command controls, and family grids
now have strict finite and native-type checks. Covariance normalization is
overflow-safe and exactly symmetric. Missing optional LaTeX text uses the
parameter name. Resume and append authenticate and semantically read the
active checkpoint without writable mappings; append then refuses because
exact continuation state is not yet saved.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Severity was HIGH because malformed normal inputs could create an undersized
or differently defined dataset that later training treated as valid science.
Evidence: generator-ingress 33/33; publication continuation 18/18; checkpoint,
run-control, seed, and request-contract gates; board self-test; full CPU suite
683 passed with 2 skipped; and final independent audit GO after three numeric
edge-case reproductions were added and fixed.
</details>

<a id="open-artifact-output-identity"></a>
<a id="closed-artifact-output-identity"></a>
### Give scientifically different emulator files different names

**High-level summary.** Two completed emulator runs could use the same output
name even when they represented different spectra, physical quantities,
selected data rows, loss modes, or source models. Saving the later run could
therefore replace a valid earlier result, and a recorded source path could
later point to different model bytes.

The saved name now starts with the output family and product and ends with a
32-character digest of the completed run. A CMB `TT` run and an `EE` run, for
example, receive different names. The same is true for the three analytic
rescaling loss modes and for different authenticated fine-tune or transfer
sources.

**Current status.** **CLOSED.** Commit `fa1ec12` records and checks the complete
output identity and refuses every occupied result name.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The digest binds the resolved model and training
settings, executed rescaling mode, exact published training and validation
generations and row order, composition rule, and reused source pair. CMB names
also bind the exact multipole, whitening-scale, and fiducial-spectrum arrays.

The `.h5` record saves the canonical identity and digest. Rebuild checks them
and the exact `.emul` weight digest before PyTorch loads a model. A complete
pair, either lone member, a symbolic link, or an interrupted-save marker
reserves the name. During a race, the first completed writer wins.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Landed on `main` as `fa1ec12`. Evidence: 610/610 CPU tests passed with one
optional skip; the registered gate passed 31 scientific-name witnesses and 19
save/publication witnesses; the CMB identity gate passed; board list and dry
run accepted the new gate; the permanent-note guard passed against the exact
commit; and two independent exact-diff reviews returned GO.
</details>

<a id="open-padded-head-identity"></a>
<a id="closed-padded-head-identity"></a>
### Stop artificial padded values from mixing with physical bins

**High-level summary.** CNN and Transformer heads use artificial positions so
physical groups with different lengths can share one rectangular tensor. The
old saved representation could not prove which positions were physical, and
some model operations could turn an artificial zero into a nonzero value that
later influenced a real prediction.

The repaired models save the complete physical-position map and validity mask.
They reapply the mask after every operation that could revive an artificial
position, and they refuse saved models whose recorded layout does not match
the model being published or rebuilt.

**Current status.** **CLOSED.** Implementation commit `32f5b48` is on local
`main`.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Plain and template CNN and Transformer heads retain
fully masked rows, keep padding inert through convolution, FiLM, activation,
normalization, attention, projection, residual, and MLP operations, and gather
only the recorded physical positions. Save and rebuild checks bind the fixed
model buffers to the geometry record and reject older count-only structured
artifacts that cannot prove coordinate identity.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>

Evidence: 60 focused model and layout tests; 36 artifact and rebuild tests; 24
permanent-note and README contract tests; the registered 24-witness
`padded-head-identity` gate; the full CMB identity gate; an isolated
matter-power structured-head check; gate-board reconciliation; Python
compilation; two clean LaTeX builds with visual page inspection; and
independent production-path, artifact, gate, and evidence reviews with GO
decisions.

</details>

<a id="open-active-model-validation"></a>
<a id="closed-active-model-validation"></a>
### Reject invalid model settings before building the model

**High-level summary.** Model settings are now checked before the program
opens training files, selects an accelerator, or creates neural-network
layers. Values with the wrong type, impossible sizes, nonfinite numbers, and
unsafe output activations therefore stop with a message that names the exact
setting the user must correct.

The same rules are repeated inside the public model constructors. A caller
that builds a model directly cannot bypass the configuration checks by
skipping the normal experiment setup.

**Current status.** **CLOSED.** Implementation commit `08172db` is on local
`main`.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The active MLP, CNN, or Transformer configuration
must use exact Boolean and integer values, finite positive scales, compatible
widths, and supported normalization and compilation modes. The correction
head uses a safe activation after its zero-initialized layer, while ReLU in
the earlier trunk remains supported. Settings for inactive model families do
not block a valid run, and an MLP with no residual blocks remains a valid
linear trunk.

**What is missing.** Nothing for this ticket. The separate mathematical
gradient problem at the exact origin remains recorded in
[Preserve the power activation gradient at zero](#open-power-zero-gradient).

<details><summary>Technical record for development tools</summary>

Evidence: 19/19 focused active-model tests; 546/546 developer tests in fresh
model/science and subprocess/policy interpreters; full gate-board self-test;
optimized-Python constructor witnesses; permanent-note guard and style tests;
two successful LaTeX builds and visual inspection of the changed guide pages;
and independent source and evidence reviews with GO decisions.

</details>

### Calculate sigma-eight at the conventional physical radius

**High-level summary.** The matter-power bridge previously used a literal
8-Mpc radius even though its saved wavenumbers use inverse megaparsecs. For a
cosmology with `h = 0.64`, it could therefore label a result near one as
sigma-eight when the conventional calculation is near 0.64.

The bridge now uses `R = 8/h` Mpc, requires the exact stored redshift zero,
and refuses a wavenumber grid whose measured tails or numerical resolution do
not support the integral. Cobaya supplies `H0` for this derived result without
adding that dependency to unrelated matter-power requests.

**Current status.** **CLOSED.** Implementation commit `3134cd5` and the
separate permanent-note administration commit `ee43ec0` are on local `main`.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The helper integrates the linear spectrum in
float64, checks an analytic known answer, and independently protects the
missing-tail, interlaced-recalculation, and largest-panel limits. The public
calculation publishes nothing until the complete result passes. The CAMB
reference stays inside the requested wavenumber range and agrees within the
declared 0.2 percent limit.

**What is missing.** Nothing for sigma-eight. General matter-power request
domains, interpolation, extrapolation, and saved calibration ranges remain in
[Reject matter-power requests outside calibrated ranges](#open-mps-serving-domain).

<details><summary>Technical record for development tools</summary>

Evidence: 515/515 developer tests; 28/28 focused sigma-eight, publication, and
README-inventory tests; every focused matter-power adapter check; direct CAMB
values `0.828513779` and `0.827662255` with relative difference `0.00102883`;
61/61 permanent-note and role tests; exact eleven-note guard; and independent
scientific, Cobaya-routing, and test/documentation reviews with final GO.

The full matter-power identity gate still stops before its adapter leg because
its older authenticated generator failure-mask fixture is absent. The full
smoke gate was not run because it generates data and trains two models; its
CAMB reference calculation was run directly. Those existing workstation
obligations remain in their own open validation tickets.

</details>

<a id="open-pce-strictness"></a>
<a id="closed-pce-strictness"></a>
### Stop the polynomial emulator from saving a fit that failed its accuracy limit

**High-level summary.** The polynomial emulator could previously keep its
first output pattern after every attempted pattern missed the accuracy limit.
It could then save a finite-looking base even though its own leave-one-out
check had rejected that base.

The fit now judges the input bounds, coefficients, and complete matrix in the
same 32-bit number format used after saving. A failing output pattern is
removed and the smaller matrix is checked again; no emulator is created when
no pattern remains.

**Current status.** **CLOSED.** Implementation commit `dd07caa` and the
separate permanent-note commit `aaac2d7` are on local `main`.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Inputs, fit limits, target variance, support
indices, leverage, coefficients, and every retained accuracy score must be
valid and finite. The selector cannot reuse a polynomial term or fit as many
coefficients as training rows. Equality with `loo_max` fails. The final
multi-output matrix is checked in its saved form, and a rejected output pattern
is left for the neural refiner instead of being forced into the polynomial
base.

The artifact still saves the six polynomial arrays `lo`, `hi`, `multi_index`,
`C`, `Vk`, and `Ybar`. Accuracy scores are checked and reported before saving;
they are not stored in those six arrays. An older artifact therefore cannot be
certified retroactively from its polynomial arrays alone.

**What is missing.** Nothing for strict fit acceptance. The separate Medium
ticket [Refuse polynomial-emulator requests outside the fitted parameter
range](#open-pce-domain-enforcement) owns the future serving-domain option.
Configured GPU training remains in the existing workstation validation
tickets.

<details><summary>Technical record for development tools</summary>

Evidence: 12/12 strict-PCE tests; 28/28 training-behavior tests; 527/527 full
CPU tests; gate-board self-test `ALL PASS`; scalar, CMB, background, and
matter-power NPCE checks with save/rebuild identity; Python compilation and
diff checks; 61/61 permanent-note and role-contract tests; both note guards;
and independent code, test, and documentation reviews with final GO.

The strict witnesses cover no-mode refusal without an artifact, unique and
non-saturated support, nonfinite values, equality at the limit, input-bound
rounding, coefficient cancellation, multi-output matrix rounding, removal of
only the failing joint mode, the unchanged six-array state, and exact finite
training-size sweep results.

</details>

### Refuse invalid values at every public prediction boundary

**High-level summary.** A saved emulator could previously ignore the output
transformation recorded during training. Public prediction also trusted
several intermediate arrays, so a Boolean, nonfinite number, wrong width, or
broadcastable matter-power row could reach a likelihood as an apparently
valid scientific result.

The public reader now serves only artifacts that explicitly record the
supported untransformed target. Inputs, encoded parameters, model outputs,
decoded values, CMB amplitude calculations, and adapter arithmetic are checked
before any result is published.

**Current status.** **CLOSED.** Commit `6c21155` is on `main`.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Missing, mistyped, or transformed `rescale` facts
stop before geometry or model loading. Every emulator family requires exact
finite prediction shapes. Background distances reject nonpositive Hubble
rates, matter-power composition rejects row or column broadcasting, scalar
adapters stage all outputs, and no adapter leaves a partial sampled-point
result after a later calculation fails.

**What is missing.** Nothing for this ticket. Workstation-only CUDA checks and
the older matter-power gate fixture remain recorded by their existing open
validation tickets; they do not change the completed CPU behavior or focused
adapter evidence here.

<details><summary>Technical record for development tools</summary>

Evidence: 507/507 developer tests; scalar, CMB, background, cosmic-shear, and
transfer identity gates; focused matter-power adapter assembly; a real-file
rescale refusal and bypass mutation; and the finite wrong-vector witness with
maximum absolute error `28.236`. The broad finite-value gate passed every CPU
arm and reported its mandatory CUDA mirror unavailable. Three independent
reviews returned GO after the Syren pre-composition shape fix and the
production-coupled test expansion.

</details>

### Keep generated datasets complete through training

**High-level summary.** A generator previously wrote related files at
different moments, so resume or append work could combine parameter rows,
payloads, axes, or failure flags that did not belong to one completed result.
Training also opened familiar flat filenames without proving that every file
came from the same generation.

**Current status.** **CLOSED.** Fresh and resumed work stays private until one
complete read-only generation is selected. Each YAML parameter filename now
finds one authenticated training or validation generation, and no mutable
flat-file fallback remains.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Generator requests and member lists are bound before
work begins. Rank-zero refusals reach every MPI worker, all writers close before
publication, and a compare-and-swap prevents stale resume work from replacing a
newer result. Cocoa pins train and validation separately and saves their member
digests, cuts, split seed, and exact staged disk-row order.

**What is missing.** Exact append, recovery of the first interrupted private
draft, persisted sampler state, and old-generation cleanup remain Medium new
functionality under [Continue generated datasets exactly and manage old
generations](#open-dataset-continuation-features). These operations fail closed
and do not change an earlier active generation.

<details><summary>Technical record for development tools</summary>
Landed on `main` as `fa8f170`. Evidence: 478/478 AI tests; 76/76 focused
dataset-publication witnesses; validation-board self-test and dry-run; exact
permanent-note guard; and independent adversarial GO after resident,
disk-backed, scalar, all-row, and wrong-permutation staging checks.
</details>

### Keep failed physics rows out of training datasets

**High-level summary.** A failed physics calculation left a finite zero vector
with the expected shape. Training could therefore mistake the placeholder for
a real cosmology and learn scientifically false behavior.

**Current status.** **CLOSED.** A full data-vector generation cannot publish
while any row is marked failed. Staging also requires the authenticated mask
and removes failed rows before cuts, seeded selection, and pool-size counting.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Only literal `0` and `1` mask rows are accepted, and
the mask length must equal the parameter and payload row counts. A requested
training size cannot be filled with a failed row. The saved row fingerprint is
checked against the disk rows that the loader actually addresses.

**What is missing.** Automatic reproducible replacement of failed rows remains
Medium new functionality under [Retry failed generator rows
reproducibly](#open-generator-failure-retry).

<details><summary>Technical record for development tools</summary>
Landed on `main` as `fa8f170`. The same 478-test suite, 76-witness publication
gate, and independent staging re-audit cover this ticket.
</details>

### Stop training before it can save an unreadable emulator

**High-level summary.** Production training could finish without the
scientific record required by public prediction. It could then save an older
or incomplete file that the same library immediately refused to reopen.

**Current status.** **CLOSED.** Production now checks both dataset records
before expensive setup and writes one readable current format.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Training validates the training and validation
facts before choosing a device, opening a warm-start or transfer artifact, or
constructing the experiment. The exact approved text is retained through
staging. Saving requires that text and the reader-required training and model
instructions, writes schema 3 only, and refuses invalid input before reading
model state or changing output files.

**What is missing.** Nothing for this production ticket. The real-workstation
[gate-fixture checks](#open-schema-v3-gate-fixtures) remain open until their
datasets are regenerated with producer-authored facts. Complete arbitrary
constructor coverage is closed in the
[saved-recipe ticket](#closed-artifact-recipe-totality).

<details><summary>Technical record for development tools</summary>
Landed as `0fe2067`, followed by the Architect-only permanent rule in
`b6c7afd`. Complete arbitrary constructor coverage is now closed in the
[saved-recipe ticket](#closed-artifact-recipe-totality). Evidence: 446/446
project tests; 17/17 focused save and refusal tests; board self-test;
parameter-table, geometry, transfer, scalar, background, CMB, cosmic-shear
adapter, and policy identity checks; and final Red Team GO after the
explicit-null recipe reproduction refused before either output file was
created. CUDA-only compilation and the real cs16/cs8 CoCoA
deployment remain unavailable and are not counted as passes.
</details>

### Publish and load each saved emulator as one authenticated pair

**High-level summary.** A saved emulator has one learned-weights file and one
scientific-record file. The two files previously had no shared fingerprint,
so a crash or file swap could join plausible but unrelated files. Loading the
weights also did not explicitly restrict PyTorch to tensor data.

**Current status.** **CLOSED.** The two files now identify one another and are
checked before model construction.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Both files are staged and checked before their
final names change. The scientific record stores a shared identifier and the
exact SHA-256 fingerprint of the weights. Ordinary failures restore the
preceding pair, an interruption leaves a refusal marker, and concurrent
writers cannot erase one another's completed pair. Rebuild refuses a swap or
malformed declaration before model construction and uses an explicit
tensor-only PyTorch load. Warm-start obtains its settings from the same checked
HDF5 file opening.

**What is missing.** Nothing for this ticket. Output-name collisions and
complete saved-recipe coverage remain separate Open tickets.

<details><summary>Technical record for development tools</summary>
Landed as `9711160de57f54546b0ee675416665131869b13c`. Evidence: 429/429
developer tests; 13/13 focused artifact-pair tests; artifact-composition
acceptance PASS; compile-recipe CPU controls PASS; Python compilation and
whitespace checks PASS; CUDA-only compile evidence unavailable on this Mac;
two independent reviews GO.
</details>

### Authenticate fixed facts in the artifact and adapter chain — Unit 84

**High-level summary.** Saved fixed scientific settings needed to remain
consistent from the training artifact to the Cobaya adapter. Unit 84 added the
first half of that shared authentication with Unit 85.

**Current status.** **CLOSED.** The joint 84/85 change is on main.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The artifact and adapter use the audited fixed-facts
path instead of rebuilding those values independently.

**What is missing.** Nothing in the production change. The accepted fixture
follow-up remains [an open gate-fixture ticket](#open-schema-v3-gate-fixtures).

<details><summary>Technical record for development tools</summary>
Committed with Unit 85 as `d3b9289`; joint audit is recorded in
`ai/notes/gates-and-board.md`. The historical schema-v2 fixture rider was sent
as mailbox 0161 and is not treated as hidden work in this closed item.
</details>

### Authenticate fixed facts in the artifact and adapter chain — Unit 85

**High-level summary.** Unit 85 completed the adapter half of the same
fixed-facts change as Unit 84. Treating the pair as one change keeps the saved
scientific settings and inference settings aligned.

**Current status.** **CLOSED.** The joint audit accepted both halves.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Unit 85 shares the same code version and evidence
record as Unit 84.

**What is missing.** Nothing in the production change. The accepted fixture
follow-up remains [an open gate-fixture ticket](#open-schema-v3-gate-fixtures).

<details><summary>Technical record for development tools</summary>
Committed as `d3b9289`; see the Units 84+85 audit in
`ai/notes/gates-and-board.md`.
</details>

### Save and verify an artifact's composition mode — Unit 96

**High-level summary.** A reader could infer plain, neural-PCE, or transfer
behavior from which HDF5 groups happened to exist. Schema-v3 artifacts now
declare the native composition mode and refined state, and the reader checks
that declaration against the exact payload and resolved YAML before loading
weights.

**Current status.** **CLOSED.** The composition declaration is enforced.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Writer, rebuild, inference, and warm-start agree on
the valid plain, NPCE, and transfer rows and reject presence-only artifacts.

**What is missing.** The real-dump confirmation is in
[the workstation ticket](#open-workstation-debt). The schema-v3 smoke-fixture
repair is tracked separately in [the gate-fixture ticket](#open-schema-v3-gate-fixtures).

<details><summary>Technical record for development tools</summary>
Committed as `3d47318`; four valid rows, 30/30 forged rows refused, 14/14
focused tests, 58/58 full tests, identity gates, board checks, compilation,
diff, and three reviews passed.
</details>

### Authenticate the Grid2D constant mask — Unit 96 rider

**High-level summary.** A saved Grid2D model could previously carry a constant
mask without a value that proved the ordered mask was unchanged. Saves now
record its SHA-256 value for the main geometry and transfer base, and rebuild
checks it before creating the model.

**Current status.** **CLOSED.** The one-surface mask check is enforced.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Missing declarations, changed order, and mask data
on non-Grid2D artifacts refuse.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Ten of ten mask tests and all seven MPS identity results passed; moved-pin,
count-only, and validator-bypass mutations failed as intended.
</details>

### Enforce boundary and interior support policy — Unit 94

**High-level summary.** Generated parameter samples needed one explicit rule
for points near the allowed boundary and points in the interior. The accepted
change enforces that policy and supplies the prerequisite used by Unit 8.

**Current status.** **CLOSED.** The current-main version is `f046085`.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The old candidate was ported and rechecked against
current main.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Current witness 12/12 passed and all four mutation controls failed at their
named checks. Do not merge the obsolete `a0a03a9` branch again.
</details>

### Validate each generated row before marking it successful — Unit 56

**High-level summary.** A generated row could be marked successful before all
serial, MPI, resumed, dtype, shape, finiteness, and byte-readback checks agreed.
Every path now uses one predicate and clears the failure flag only after the
written bytes are read back exactly.

**Current status.** **CLOSED.** All row-writing paths use the shared check.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Corrupt or wrong-dtype resumed rows refuse without
changing payload bytes, timestamps, or the failure mask.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Landed and pushed as `e885a8d`; 24/24 focused tests, 292/292 full tests,
compilation, diff, permanent-note guard, and focused review passed.
</details>

### Centralize background quantity and unit validation — Production Unit 62

**High-level summary.** Background quantity and unit pairs were checked in
several places and could disagree. One registry now controls configuration,
geometry, rebuild, and the Cobaya background adapter.

**Current status.** **CLOSED.** All four paths share the registry.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Wrong pairs and non-string, nonfinite, Boolean, or
quoted offsets refuse before save or inference.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Landed and pushed as `c6fca01`; 9/9 focused tests, 260/260 full tests,
compilation, diff, and permanent-note guard passed.
</details>

### Require CPU-normalized saved model state — Units 64 and 70

**High-level summary.** Saved `.emul` state needed direct proof that it contains
a nonempty tensor dictionary and that every tensor is stored on the CPU. A
ninth independent result now checks those bytes without a load-time device
override.

**Current status.** **CLOSED.** The local saved-state rule is enforced.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** An inert compile-default claim was removed, while
the activation-default check and CPU-state refusal cases remain active.

**What is missing.** Nothing for this ticket. The CUDA, CosmoLike, deployment
dump, and `.cpu()` removal run is tracked in
[the workstation ticket](#open-workstation-debt).

<details><summary>Technical record for development tools</summary>
Landed and pushed as `fb5302e`; 58/58 tests, board self-test/list,
compilation, diff, helper refusal cases, and two reviews passed.
</details>

### Preserve Grid2D row identity during staging — 25M-32/33

**High-level summary.** Grid2D staging could lose the generator's seeded row
order while moving through raw, base, parameter, data-vector, and index arrays.
Resident and memory-mapped paths now preserve one exact row identity and check
all row counts before allocating transformed targets.

**Current status.** **CLOSED.** Both staging modes use the accepted order.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** N-minus-one and N-plus-one inputs refuse, while the
exact N-row input passes.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Landed and pushed as `c688489`; 278/278 tests, seven MPS identity results,
row-count witnesses, order mutation, compilation, diff, note style/guard, and
independent review passed.
</details>

### Authenticate optimization-study identity — Unit 53 repair

**High-level summary.** An optimization study could reuse results created with
different scientific inputs, family choices, or implementation rules. One
manifest now fixes that identity before workers start and prevents failed or
stale trials from becoming the winner.

**Current status.** **CLOSED.** Local study identity and worker rules are
enforced.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Workers rebuild the manifest identity, loaded old
studies refuse, and the manifest-owned default is queued once.

**What is missing.** Nothing for this ticket. The real Optuna journal run is
tracked in [the workstation ticket](#open-workstation-debt).

<details><summary>Technical record for development tools</summary>
CoCoA Torch witness 34/34, ten critical/coupling mutations, shipped-threshold
canonicalization, exact `n_theta` environment drift, compilation, diff, and
two audits passed.
</details>

### Repair the generator-ranges gate

**High-level summary.** The range gate could miss an old header format because
GetDist might accept or reject comment rows before the intended assertion ran.
It now checks the producer-owned rows-only sidecar directly.

**Current status.** **CLOSED.** The intended sidecar rule is reached.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The exact retired header and decimal-format change
are both caught.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
See `ai/notes/gates-and-board.md`, Gate-integrity pair recovery audit.
</details>

### Repair the cross-family transfer-refusal gate

**High-level summary.** The cross-family transfer check failed early because
its fixture omitted ordinary required data. The fixture now reaches the actual
rule that forbids a transfer base from the wrong family.

**Current status.** **CLOSED.** The named scientific refusal is tested.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Removing `n_train` or `n_val` now fails only the
intended early-data control instead of masking the cross-family result.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
See `ai/notes/gates-and-board.md`, Gate-integrity pair recovery audit.
</details>

### Resolve parameter names for numbered chain files

**High-level summary.** A file such as `chain.1.txt` could fail to find the
shared `chain.paramnames` declaration. One resolver now applies the numbered
root fallback in ordinary, fixed-facts, and scalar staging.

**Current status.** **CLOSED.** All affected staging paths share the resolver.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The exact numbered file, plain table, and dotted
nonnumeric stem rules are tested before data loading.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Six of six focused tests and `stage-ram` passed; deleting the numeric-root
fallback failed the focused witness.
</details>

### Retire a stale rebase ticket — Unit 90

**High-level summary.** Unit 90 appeared unfinished even though its accepted
implementation was already part of main. The backlog entry was reconciled
against Git history instead of merging the same work again.

**Current status.** **CLOSED.** No code change was required.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Commit `50f1c63` is an ancestor of main through
`ce99f87`.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Evidence is the Git ancestry check and the Unit 90 verdict in
`ai/notes/gates-and-board.md`.
</details>

### Validate the CMB covariance package — Unit 13

**High-level summary.** CMB covariance generation needed explicit wiring and
independent failure checks for each validator. The accepted package combines
the scientific calculation with the command path that runs it.

**Current status.** **CLOSED.** The CPU package and wiring are on main.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Five CPU witnesses pass and four independent
validator-removal changes fail their named program checks.

**What is missing.** Nothing for this ticket. Torch CMB identity and real-CAMB
byte identity are tracked in [the workstation ticket](#open-workstation-debt).

<details><summary>Technical record for development tools</summary>
Substance commit `2fd8a9d` plus wiring commit `7583019`; durable owner is
`ai/notes/families-scalar-cmb.md`.
</details>

### Recheck sampled parameter order before loading artifact weights

**High-level summary.** A forged schema-v3 record and matching sidecar could
agree with each other while disagreeing with the rebuilt input geometry. The
reader now compares the sampled-name order with that independent geometry
before calling `torch.load`.

**Current status.** **CLOSED.** The coordinated forgery is refused early.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The valid control advances, while bypassing the new
comparison reaches weight loading and makes the focused test fail.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
See `ai/notes/gates-and-board.md`, Rebuild-time fixed-facts name audit.
</details>

### Use an Ollama model as the Implementer

**High-level summary.** The watcher previously built both the Architect and
Implementer commands around Anthropic's Claude service. A user who could run a
capable open-weight coding model locally still had to spend Claude allowance
for the token-heavy implementation work.

The Architect and Implementer providers are now independent. The Architect
remains on Claude, while `--implementer-provider ollama` sends the Implementer
role to a named Ollama model through Ollama's supported headless coding
integration.

**Current status.** **CLOSED.** This was accepted as **LOW NEW
FUNCTIONALITY**. The default remains Claude, so existing watcher commands do
not change behavior.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The provider and model are separate command-line
choices. Ollama receives the same isolated Implementer worktree, checkpoint
hook, Architect directive, and evidence requirements. `--ping` checks the
Ollama model when selected, and the terminal names each role's provider.

**What is missing.** Nothing for this ticket. The workstation running the
watch must install Ollama, start its service, and download a coding model with
enough context for the requested ticket.

<details><summary>Technical record for development tools</summary>

Primary code: `ai/tools/mailbox_daemon.py::build_agent_commands` and
`check_provider_connectivity`. Focused unit tests replace all provider
programs, while `tools_mailbox_daemon_role_models_repro.py` verifies CLI
wiring and kills mutations that ignore the provider or model choice.

</details>

## Mailbox and watcher behavior

### Start every role turn with an empty provider conversation

**High-level summary.** A user asked whether the Architect, Implementer, and
Red Team should compact their conversations after a ticket closes so an old
ticket cannot fill the context window during the next one.

The watcher already starts a separate provider conversation for every mailbox
turn, but the launch commands did not explicitly forbid saving those sessions.
The intended fresh-context boundary therefore depended on nobody later adding
a resume option.

**Current status.** **CLOSED.** Claude dispatches are now explicitly
non-persistent and Sol dispatches are explicitly ephemeral. No separate paid
compaction turn is created.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** A focused reproduction checks both Claude routes
and the Sol route, and refuses a future command that resumes an earlier
provider session. The tool guide explains that the context limits apply only
inside one unusually long role turn.

**What is missing.** Nothing for cross-ticket context isolation. A single
role turn can still compact before it finishes if that turn alone reaches its
configured context limit; increasing the limit or splitting an oversized
ticket addresses that different case.

<details><summary>Technical record for development tools</summary>

`build_agent_commands()` supplies Claude `--no-session-persistence` and Sol
`--ephemeral`. `arm_each_dispatch_starts_fresh` proves that neither Claude
route contains `--continue` or `--resume` and that Sol does not use
`codex exec resume`.

Landed as `5b3f84f5a5f511064de9bc0ce56f50ae39b5f8d2` after the focused
command reproduction, 91 mailbox tests, and an independent adversarial GO.

</details>

### Stop cleanly when an AI account runs out of tokens

**High-level summary.** Claude or Sol can exhaust an account allowance during
a live watch. The daemon formerly saved the request but kept polling without
telling the user that more credits were required.

**Current status.** **CLOSED.** A verified provider account-limit message now
stops the watch with `Error: Architect is out of tokens`, `Error: Implementer
is out of tokens`, or `Error: Sol is out of tokens`.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The exact request is preserved in `failed/`, the
relay log and role worktree remain untouched, another role that already
started is allowed to finish, and no later request starts in that watch pass.
Transient rate limits and context-size failures are not mislabeled as account
exhaustion.

**What is missing.** Nothing for safe stop-and-preserve recovery. Retrying is
a user action after credits are restored, because automatically resetting or
committing a partially edited Implementer worktree could lose work.

<details><summary>Technical record for development tools</summary>

The staleness reproduction covers all three public role names, nearby false
positives, exact inode and byte preservation, no invented timeout history, a
waiting same-role request, and a parallel Sol job that finishes first.

</details>

<a id="closed-user-main-primary-sync"></a>
### Let a clean user update on main reach every AI worktree

**High-level summary.** An ordinary commit or pull in the user's clean main
folder could leave the three saved AI worktrees one commit behind. The next
watcher command then refused to start because a user commit has no internal
ticket-landing receipt.

**Current status.** **CLOSED.** A command launched from the clean user main
folder now recognizes that exact commit and advances every clean idle AI
worktree to it.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The fallback works only when no ticket or candidate
is active, the user checkout is attached to `main`, and its files and index
are clean at the exact main commit. Dirty work, divergent history, malformed
landing requests, and moving only the main reference remain refusals. The
existing baseline helper preserves active candidate work while aligning idle
Architect, Implementer, and Red Team folders.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Committed as `35f39b4`. The disposable Git regression accepts a real user-main
commit and refuses an Implementer-authored reference-only move. Fifty-nine
unit and role-contract tests, five affected recovery arms, the complete
staleness runtime and 9/9 mutation checks, compilation, and whitespace checks
passed. The live `--once` command advanced all three role branches from
`957afc4` to `35f39b4`, reported an empty mailbox, and exited zero.
</details>

<a id="closed-clean-all-ai-worktrees"></a>
### Remove every AI-created worktree and branch on explicit request

**High-level summary.** Old AI sessions can leave enough worktrees, branches,
and mailbox history that the daemon cannot safely decide which folder is the
current one. The explicit `--clean-all` command now lets the user discard all
of that local AI work, including dirty files and unmerged commits, without
deleting ordinary user branches or worktrees.

**Current status.** **CLOSED.** Cleanup runs before primary-folder selection,
so it remains usable when old mailbox histories make `--once` ambiguous.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The command refuses a live mailbox process, removes
registered and abandoned AI folders, deletes local `claude/*`, `codex/*`, and
legacy `worktree-agent-*` branches, and leaves remote branch records, tags,
stashes, and non-AI Git work intact. New Claude role branches use `claude/*`;
new Sol branches use `codex/*`. Cleanup never runs automatically.

**What is missing.** Nothing for this ticket. The user may now run
`python3 ai/tools/mailbox_daemon.py --clean-all` from the main repository
folder when the old AI work is no longer needed.

<details><summary>Technical record for development tools</summary>
Landed and pushed as `957afc4`. Evidence: 138 focused tests, a fresh-role
bootstrap check, exact help/README parity, compilation and whitespace checks,
and one adversarial protected-policy review all passed. The destructive command
was tested only in disposable repositories and was not run in the live clone.
</details>

<a id="closed-provider-connectivity-ping"></a>
### Check whether Claude and Sol can answer before starting work

**High-level summary.** The former ping command only placed a message in the
mailbox. It could not tell the user whether Claude or Sol was logged in and
able to answer. Bare `--ping` now makes one small live request to each service,
while `--ping --skip-redteam` checks Claude without starting Sol.

**Current status.** **CLOSED.** The direct connection check is on `main` and
has been pushed to GitHub.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Each requested service must return the exact
nonce-bearing reply within two minutes. Claude runs without tools or a saved
session; Sol runs read-only in an empty temporary folder. Failure returns a
nonzero status, and the command does not create worktrees, mailbox messages,
backlog changes, or ticket cycles.

**What is missing.** Nothing for this ticket. The user can run `--ping` for
Claude and Sol or add `--skip-redteam` for Claude alone.

<details><summary>Technical record for development tools</summary>

Landed and pushed as `24e7888`. Evidence: 107 focused unit and contract tests;
18/18 two-role runtime checks plus 16/16 mutations; 10/10 dead-mailbox runtime
checks plus 7/7 mutations; dry-run zero-write checks; compilation and diff
checks; and one exact staged adversarial review returning GO. Provider tests
used fake subprocesses, so validation spent no Claude or Sol credits. The
source was simplified before review; its remaining size covers two distinct
provider CLIs and their isolated response channels.

</details>

<a id="closed-role-contract-coverage"></a>
### Make the protected role contract cover every stable security authority

**High-level summary.** The protected YAML named only part of the authority
used by the AI tools. A later Python edit could therefore weaken a guard,
change a saved role worktree, or admit a control file without making the
protected contract visibly change.

**Current status.** **CLOSED.** Contract schema 2 is committed and pushed on
`main` as `96766d6`.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The YAML now records the exact eleven notes, all
three role files, both guard files, all nine trusted tools, the Architect
backlog, candidate-forbidden Git and mailbox paths, size limits, and the exact
three role worktrees. Small consumers read those values directly. The daemon
independently compares them with the exact admission sets, live backlog and
mailbox paths, cleanup prefixes, and trusted files that it actually uses.

The permanent-note guard derives its note folder and census from the contract
and always checks both bootstrap guard files. A force-added backlog, mailbox
message, relay log, Git-control file, or role-directory file is refused from
an Implementer candidate.

**What is missing.** Nothing for this ticket. The router's obsolete primary
state schema is a separate Critical ticket and remains Open.

<details><summary>Technical record for development tools</summary>

One adversarial review returned NO-GO because the first draft compared the
YAML with a third set of literals instead of the exact live enforcement
values. The Architect corrected that binding and, as required, did not request
a second review. Tests now mutate the live admission set, trusted-tool set,
cleanup action and prefixes, backlog path, mailbox path, and relay path; every
drift is refused.

Evidence: 135 focused unit and role-contract tests; 6/6 backlog-bundle runtime
checks; 9/9 staleness mutations; the complete disposable primary-worktree
runtime and mutation matrix; compilation and whitespace checks; and an exact
base/candidate change-guard check. The non-test change exceeded the 4,000
character warning because this was one atomic schema migration across the
protected YAML, its reader, its consumers, and the matching role guidance.
Splitting those pieces would temporarily leave two disagreeing sources of
authority, which is the Critical defect this ticket removes.

</details>

### Require one adversarial review of protected policy changes

**High-level summary.** The Architect and Red Team role files and the eleven
permanent notes control how later work is planned. When Red Team is enabled,
the Architect now shows it the exact proposed wording once so that an
unnecessary, oversized, or contradictory rule change receives an independent
challenge.

**Current status.** **CLOSED.** Red Team gives one read-only advisory response.
The Architect then makes the final decision. There is no revision loop, second
review, veto, or post-landing policy review.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** `MAILBOX-TICKET: policy` names the cycle-free
review, including during fix-only maintenance. The two role files share the
Architect-only protected landing path with the permanent notes. A protected
administration turn also refuses to report “no change” after leaving an
uncommitted protected edit.

**What is missing.** Nothing for this ticket. With Red Team disabled, the
Architect records that the independent review was unavailable and applies the
same narrow protected-file checks.

<details><summary>Technical record for development tools</summary>
The one required adversarial review recommended a smaller design. The accepted
implementation uses one ordinary read-only mailbox response and the existing
protected landing path instead of a review journal, reusable approval token,
private proposal reference, or second decision round. Evidence: 80 focused
unit and role-contract tests; 38/38 disposable-worktree runtime cases,
including refusal of an uncommitted protected-role edit; and 56/56 deliberate
safeguard-removal mutations, all armed and refused for the intended reason.
</details>

### Pause long Implementer work for an Architect complexity review

**High-level summary.** An Implementer could previously spend several hours
expanding one repair before the Architect saw that the approach had become too
complicated. The watcher now asks the Implementer to pause after 90 minutes,
save coherent progress, and explain the size, remaining work, elapsed time,
and complexity of the approach.

**Current status.** **CLOSED.** The pause is a progress review inside the same
ticket and cycle. It cannot be accepted as a finished candidate or landed.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The main Implementer receives one instruction at
the next completed tool action; helper agents cannot consume it. The Architect
may authorize another bounded period or replace the plan. A later 120-minute
timeout remains available for an AI process that stops responding.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
The focused hook and role suites, temporary-repository refusal witness,
staleness, ticket-cycle, landing-debt, permanent-note, inventory, compilation,
and whitespace checks passed. The Red Team reviewed the reduced design and
the protected wording and returned GO.
</details>

### Require a checkpoint decision before implementation resumes

**High-level summary.** A 90-minute pause could previously return to the
Implementer without one explicit Architect decision. A checkpoint with no new
commit also had no immutable candidate for the Architect to inspect.

**Current status.** **CLOSED.** Every timed checkpoint now needs a new clean
commit and one fresh same-cycle, same-mode GO or NO-GO handoff before work can
resume.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Malformed or contradictory checkpoint state is
refused before the Architect starts. Checkpoint prompts omit ordinary landing
instructions, a checkpoint cannot send landing GO, and conflicting fresh
outputs are parked together rather than accepting only the convenient one.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Landed and pushed as `4e21b6f`. Evidence: 148 unit and contract tests; six
live checkpoint branches; six safeguard-removal mutations; staleness,
ticket-cycle, and landing-debt suites; compilation and whitespace checks; and
two independent adversarial reviews GO. The approximately 6.4k-character
production change exceeds the 4,000-character warning with a recorded
Critical exception: prompt authority, immutable checkpoint identity, exact
decision binding, and atomic output refusal had to close together so no
intermediate commit left the pause bypassable. No schema, saved phase, history
scan, framework, or production file was added.
</details>

### Keep GitHub commit messages readable

**High-level summary.** Landing commits previously replaced the Implementer's
explanation with an internal ticket label. A human reading the GitHub history
could not learn what changed, why it changed, or which checks passed without
opening the code diff and the tracked backlog.

**Current status.** **CLOSED.** Manual AI commits and mailbox commits now
follow the same human-first writing rule.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Every AI-authored commit needs a concrete subject
and a short Markdown body that explains the observed problem, the saved
change and its boundary, and exact evidence. The watcher preserves the exact
Architect-approved UTF-8 message in the landing commit, adds only two reserved
recovery lines, and refuses ambiguous or altered recovery messages.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Implementation commit `861acd5`; permanent-rule commit `3e22a1b`. Evidence:
429/429 project tests; 37/37 watcher runtime scenarios; 34/34 role-contract
tests; focused raw-message and crash-recovery reproduction; compilation and
whitespace checks; independent code and permanent-rule reviews GO.
</details>

### Keep mailbox dry runs read-only

**High-level summary.** A dry run could move a malformed or placeholder
message even though the user asked only to preview the action. Dry-run mode now
leaves the exact pending file in place and creates no failed record.

**Current status.** **CLOSED.** Preview and real dispatch have separate file
behavior.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Only a real dispatch claims a pending file into the
work-in-progress folder.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
See `ai/notes/gates-and-board.md`, Current-daemon transport safety audit.
</details>

### Use readable sentence-case terminal output

**High-level summary.** Daemon output used dense separators and all-capital
phrasing that was difficult to scan. User-facing lines now use sentence case
and semicolons while preserving exact protocol and acronym text.

**Current status.** **CLOSED.** Runtime and README quotations agree.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Separator and capitalization changes are covered by
focused refusal controls.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Focused output checks 8/8 and daemon reproduction 8/8 passed; separator and
all-caps mutations failed.
</details>

### Do not require a reply to an explicit terminal message

**High-level summary.** Every inbound mailbox message formerly demanded a
reply, including one that explicitly said the conversation was finished. An
exact terminal/no-reply message is now exempt, while any ambiguity still
requires an outbound response.

**Current status.** **CLOSED.** The single exception is enforced.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Ordinary notes-first prompts retain the required
reply and both parts of the wording are mutation-tested.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Four prompt/surface regressions and the eight-arm daemon reproduction passed;
removing the ordinary rule or terminal exception failed its focused check.
</details>

### Keep claimed mailbox work current and publish outcomes safely

**High-level summary.** A claimed message could become ambiguous when newer
work arrived, a timeout occurred, or another process touched its archive path.
Each dispatch now records one current-state snapshot and publishes only the
exact claimed file to done or failed.

**Current status.** **CLOSED.** Later work stays blocked until the claimed item
has a truthful outcome.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Timeout history, `--once` propagation, exact file
identity, line endings, dry-run behavior, and hostile history refusal are
covered.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Focused runtime 18/18, nine source mutations, preserved daemon/output/preamble
suites, and two independent reviews passed.
</details>

### Complete the combined daemon repair program

**High-level summary.** Several related daemon repairs were once tracked by one
umbrella line in addition to their individual tickets. This roll-up confirms
that recovery, prompts, output, archive handling, watcher warnings, fix-only,
safe stopping, and landing-debt behavior all have accepted child records.

**Current status.** **CLOSED.** This is a summary, not a separate code defect.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Each child appears as its own closed ticket in this
section with focused evidence.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Source record: daemon audits through Daemon landing-debt self-correction audit
in `ai/notes/gates-and-board.md`.
</details>

### Warn when a sent message has no live watcher

**High-level summary.** A send could succeed into a mailbox that no watcher was
reading, leaving the user to assume work had started. Send and ping now name
that mailbox and any other live watched mailbox without rerouting or failing
the successful send.

**Current status.** **CLOSED.** The warning is visible and dry-run stays
read-only.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Exact owner parsing, file identity, and link refusal
keep the warning conservative.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Focused runtime 9/9, seven mutations, daemon/output/preamble/staleness suites,
and two reviews passed.
</details>

### Protect fix-only mode and classify Sol tickets

**High-level summary.** Fix-only mode could not reliably distinguish a known
repair from a new discovery, especially near the discovery-demand limit.
Public Sol requests now declare closure or discovery, and fix-only launches
only accepted closure work plus the exact internal transport ping.

**Current status.** **CLOSED.** Ambiguous actions fail before launch.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Saturated discoveries are preserved with a clear
instruction, demand-nine work keeps its classification, and one mode lock
binds child and external sends.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Focused runtime 14/14, twenty mutations, dead-mailbox, Red Team, staleness,
output, and preamble suites plus two reviews passed.
</details>

### Provide regular windows for stopping the watcher

**High-level summary.** A busy watcher gave the user only occasional idle
moments to stop it without interrupting active work. After five completed role
runs or fifteen continuously busy minutes, it stops starting work, waits for
started work, and prints a 19-to-0 Ctrl-C window.

**Current status.** **CLOSED.** The bounded stop window is implemented.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Queue bytes survive Ctrl-C, source edits prevent the
next admission, and preview/one-pass commands keep their finite behavior.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Focused runtime 9/9, seven boundary mutations, preserved daemon suites, and
the Architect/Red Team wording audit passed.
</details>

### Stop the watcher after a chosen number of cycles

**High-level summary.** The user needed a planned stopping point instead of
waiting to catch one short Ctrl-C window. Positive `--cycle N` now stops after
N completed safe windows, while zero waits until no enabled message and no
open backlog ticket remain.

**Current status.** **CLOSED.** Omitted, positive, and zero modes have distinct
tested behavior.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Waiting messages remain untouched at the positive
limit, concurrent sends receive truthful watcher status, and missing or
changing backlog files keep zero mode running rather than closing early.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Focused runtime 20/20 with 21 mutations, all related daemon/router suites,
44/44 CoCoA tests, board self-test, compilation, and diff passed.
</details>

### Request landing when uncommitted work grows too large

**High-level summary.** A long-running branch could collect too much accepted
work without asking the Architect to land it. Above 400 changed lines, the
watcher now creates one durable landing request for that continuous episode
and prints the current debt in each demand report.

**Current status.** **CLOSED.** Landing requests are deduplicated and rearm
after the branch returns to 400 lines or fewer.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Fable checks for foreign commits before a squash,
and one repository lock prevents Fable and Sol from landing concurrently.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Focused runtime 15/15 and ten mutations on both runtimes, preserved suites,
44/44 CoCoA tests, board self-test, compilation, diff, and two reviews passed.
</details>

<a id="closed-implementer-context-handoff"></a>
## Preserve exact Implementer context before a replacement session

### High-level summary

An Implementer can reach the context limit while a ticket is only partly
finished. Repository files survive, but a fresh Implementer also needs to know
what succeeded, what failed, and which rejected approach would waste time if
repeated.

The watcher now asks for one small record before automatic context replacement.
The replacement reads that exact record and checks the repository instead of
receiving a summary invented by the watcher.

### Current status

**CLOSED.** This was accepted as **LOW NEW FUNCTIONALITY**. Repeating an
unsuccessful approach wastes tokens, but the former behavior did not corrupt
scientific output or erase repository work.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

### What is already fixed

The automatic context hook asks the Implementer to name the ticket, base and
current commits, candidate status, completed work, failures, rejected
approaches, uncommitted files, next action, and work that must not be repeated.
The watcher verifies those facts against the current Implementer worktree.

The record follows the existing checkpoint path. It creates no candidate and
closes no cycle. After the Architect permits continuation, the replacement
receives the exact saved path and keeps the verified unfinished worktree.

### What is missing

Nothing for this ticket.

<details><summary>Technical record for development tools</summary>

`implementer_checkpoint_hook.py` requests the record through Claude Code's
automatic `PreCompact` hook. `mailbox_daemon.py` parses the bounded record and
verifies the cycle, base, HEAD, candidate
claim, and dirty state before reusing the existing checkpoint route. Focused
tests cover clean and dirty records, malformed fields, stale identity, exact
saved-path delivery, preservation of unfinished work, and absence of landing
instructions.

No session graph, task scheduler, supervisor, or additional role was added.
If a provider stops before creating the record, the existing out-of-token
recovery continues to preserve the request and worktree without fabricating a
summary.

</details>

## Repository organization and release hygiene

### Move study helpers into `emulator/studies/`

**High-level summary.** Four related modules used a repeated `study_` filename
prefix in the main emulator folder. They now form one `emulator.studies`
package, which makes their relationship visible without changing scientific
behavior.

**Current status.** **CLOSED.** The old flat files and compatibility duplicates
are absent.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** Production, gates, comments, and package maps use
the package namespace.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
The Unit-53 manifest witness, 44/44 tests, board import closure, compilation
and import probes, and independent named-change audit passed.
</details>

### Keep operational backlog files out of release history

**High-level summary.** Temporary backlog and mailbox records could be mixed
with the eleven durable notes and accidentally enter a release commit. The
repository now ignores operational records, tracks exactly eleven permanent
notes, and represents accepted work with one reviewed commit.

**Current status.** **CLOSED.** The v1.0beta1 hygiene prerequisite is satisfied;
no release tag was created.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** The primary-worktree reproduction force-tracks only
its disposable synthetic backlog, which also repairs the older fixture drift.

**What is missing.** Nothing for this ticket.

<details><summary>Technical record for development tools</summary>
Landed as `c91791a`; primary-worktree runtime 20/20 with mutations killed,
65/65 focused tests, permanent-note guard, tracked/ignored census, README
checks, and whitespace check passed.
</details>

### Move all AI-development support under `ai/`

**High-level summary.** Tests, notes, gates, and tools were scattered among
several root folders and old entry points. `ai/README.md` is now the single
starting point, with the four support folders only under `ai/`.

**Current status.** **CLOSED.** Paths, imports, tools, documentation, and
ignored transport state use the consolidated tree.

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**What is already fixed.** No old root directory, wrapper, duplicate entry
point, or compatibility link remains, and the documentation PDF builds from
the new paths.

**What is missing.** Nothing for this filesystem move. The real-data full-board
run is tracked in [the workstation ticket](#open-workstation-debt).

<details><summary>Technical record for development tools</summary>
Evidence: stale-path/filesystem audits, compilation/import probes, 44/44 tests,
nine tool reproductions, board list/dry-run/self-test, focused CPU gates, tool
help/status, and an 83-page PDF render. The development Mac correctly lacked
CUDA, CosmoLike, configured `$ROOTDIR`, and training dumps.
</details>

<a id="open-router-primary-schema-three"></a>
## Let the router read the current primary-worktree record

### High-level summary

The daemon creates the saved Architect worktree with state schema 3 and the
topology name `separate-role-worktrees-v1`. The router formerly expected the
retired schema and topology, so a valid fresh setup could fail before a role
read the authoritative backlog.

The source mismatch was repaired, but the ticket remained listed as Critical
and Open. The workflow also allowed that stale bookkeeping because landing did
not require the Architect to close and seal the exact ticket first.

### Current status

**Ticket type: BUG FIX.**

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**CLOSED.** Commit `864b69e2f300b44e40a270b4343f3bd495557a80`
teaches the router the current record. Commit
`9f99d1a9441771b1c5889002e3d13f4ee3d23bce` makes a Closed, sealed ticket a
prerequisite for a new landing.

### What is already fixed

The router reads the daemon's schema-3 record and retains its checks for the
repository, branch, path, topology, and file type. The daemon now refuses GO
while the ticket is still Open, preserves candidate C, and retires a rejected
same-cycle GO after a corrected GO lands, including after restart.

### What is missing

Nothing for this ticket.

<details><summary>Technical record for development tools</summary>

The closure gate is in `ai/tools/mailbox_daemon.py`; the Architect instruction
is in `.claude/FABLE_ROLE.md` and `ai/notes/conventions-and-workflow.md`.
`test_architect_go_needs_the_exact_ticket_closed` and
`arm_architect_receipt_binds_candidate_to_squash_landing` cover refusal,
candidate preservation, corrected GO, and restart cleanup.

</details>
