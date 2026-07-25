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
- [Closed tickets](#closed-tickets) — a pointer; the sections themselves live
  in [`backlog-closed.md`](backlog-closed.md)

## How to read this backlog

Only the Architect writes this file. The Implementer and Red Team may read it,
but they never edit it or replace its fingerprint — a saved SHA-256 detects an
unexpected edit before the Architect writes again. The daemon saves the sealed
ticket update in the same landing commit as the accepted fix.

The watcher counts lines beginning exactly with `- OPEN`, one for each
unfinished ticket in the index below. Each also records `BUG FIX` or `NEW
FUNCTIONALITY`, so the watcher can tell an existing defect from a feature at
the same priority. Never add a second `- OPEN` line inside a ticket.

Every ticket shows a **Red Team reopen count**, which starts at zero and never
resets, and whether another reopening is allowed. Red Team review is advisory:
the Architect may accept and commit a fix without it, and a `REOPEN` creates
later work rather than undoing the commit. On a `REOPEN` the Architect restores
the ticket to Open and adds one to the count, then later rules GO (repair) or
NO-GO. NO-GO closes the ticket permanently and sets its reopening line to
**barred by Architect NO-GO**; a genuinely different defect then needs a new
ticket. The sixth reopening automatically makes a ticket Low, so a repeated
disagreement cannot consume the queue.

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
- OPEN **MEDIUM** **BUG FIX** — [Close the remaining verified emulator audit findings](#open-emulator-audit-wave2)

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
- Audit verdict: GO to land the implementation, NO-GO to close this ticket.
  Closure still requires a CUDA workstation run with both assertion ids PASS
  and a child return code of 0. The two hard-coded-mode source mutations
  should fail there as the final proof that the check can catch the defect.

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

**CLOSED.** Column two is now `minuslogpost` and carries minus the sampler's
log probability, so the row with the larger posterior carries the smaller
number. A uniform draw evaluates no posterior and writes a neutral zero in
every row instead of a fabricated one. The derived `chi2*` column is
`2 * minuslogpost`, numerically identical to the previous `-2 * lnp` in the
tempered branch. The component README, the generator prose, and five layout
comments elsewhere now name the same column.

**Severity: MEDIUM.** A normal GetDist plot or ranking can be misleading, but
the sign does not change generated physics vectors, training, or values served
by an emulator.

### What is already fixed

The generator writes weights, sampled coordinates, and `chi2*` in a stable
table with parameter sidecars.

### What is missing

Nothing for this ticket. The known-answer test required below now exists as
`ai/tests/test_generator_posterior_column.py`: it writes a two-row chain with
the production writer, loads it with GetDist, and asserts the better-posterior
row ranks better; a companion test asserts that storing the sampler's sign
unchanged reverses that ranking.

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

**OPEN.** The function-docstring portion runs one file per commit, each proving
the change is docstring-only by comparing the two versions' abstract syntax
trees with docstrings stripped, and each landing on a green full suite.

The bar is a strict mechanical census, adopted after a user spot check found
one-line docstrings surviving a "complete" claim: every callable with
parameters carries an `Arguments:` block, every value-returning callable a
`Returns:` block, and no non-trivial body keeps a one-line docstring. Beyond
structure, the prose must leave no term of art or Python mechanic unstated —
that rule came from docstrings that were structurally correct but assumed the
reader knew what a sidecar, a struct round trip, or the bool-int subclass trap
was.

Complete and census-zero: the `emulator/` package (all forty files), the
`compute_data_vectors/` generators, the five `cobaya_theory/` adapters, and
`ai/tools/` (all twenty-five files, including the ten large mailbox part
files). Also complete: `documentation/emulator_code_guide.tex` (five stale or
missing claims repaired, the gate-board appendix split into its own document),
`emulator/README.md` Appendix D2, the move of the twenty-one family drivers
into `driver/`, the move of project instructions to `.claude/CLAUDE.md`, and
the removal of AI-workflow material from the main README, which `ai/README.md`
now solely owns.

Remaining: `ai/tests/`, and the repository-wide chronology rewrite outside
`emulator/`.


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

<a id="open-emulator-audit-wave2"></a>
## Close the remaining verified emulator audit findings

### High-level summary

A user-ordered file-by-file audit covered all 40 emulator/ files (ten
parallel reviewers, 2026-07-23; every finding below was reviewer-reported
and the fix-wave items were Architect-re-verified against the code). Wave 1
landed as commit ef2a85c: the MPS float64 loss crash, the transfer-refine
staging autograd graph, the fine-tune-sweep KeyError, the bs-vs-rows
ZeroDivisionError, the train-plan-on-val evals, the rising "step" anneal
collapse, the stale params stashes, and the deletion of the dead
ElementWeightedChi2 / NLAAmpFactoredChi2 classes. This ticket holds
everything verified but NOT yet fixed, so the next session resumes here.

### Report-only (design-sensitive; do not fix without a directive)

- losses/cmb.py _factor: ~5 host syncs / graph breaks inside the
  compiled hot loss (values correct; performance; workstation-verify).
- experiment.py NPCE fit materializes the full staged selection in RAM
  and on device, defeating memmap staging (OOM on the configurations
  staging protects).
- Structural duplication: from_config's six-fold activation block and
  four near-clone finetune branches (~250 lines); plain.py/ia.py
  six-way trunk/head stanzas; the n_tokens re-segmentation implemented
  twice (plain.py:1031 vs results.py:1063); diagnostics' four
  copy-pasted forward loops; activations' gate machinery in three
  classes.
- Parked/deliberate (do not delete): the finetune anchor machinery
  (open-finetune-anchor is the ticket); LogParamGeometry (unbuildable
  but rebuild-whitelisted); TARGET_LAWS payload tuples unread;
  scaler_policy single-value plumbing; the double init_probes at
  output.py:388/396 (workstation-owed A/B, board item).
- Pure LOW simplifications, deferred as churn: the unused min-max stats
  mode and discarded std in data_staging; blocks.py's two
  belt-and-suspenders unreachable checks; cocoa.py's twice-computed
  chains path; parameter_table's subsumed overlap check.
- plot_xi's port-caveat items (post-figure `return 0` leaks, the
  index-colored/value-labeled colorbar). Settle the authority question
  first: the training-stack note and the byte-faithful-port docstring
  disagree on the intended contract.

### Current status

**Ticket type: BUG FIX.**

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

OPEN (narrowed to report-only + deferred simplifications). Wave 1 landed
as ef2a85c; wave 2's fix candidates all landed across nine commits
(ffb9aec geometry guards; 6b126f1 MPS float64 rebuild; 7adf66f warmstart
structured-head refusal + extras eigh; 5d17e31 inference/fixed_facts/
results/model_recipe refusals; a0406f0 experiment config-validation gaps;
e48c20c anneal-const + training contracts; 2bd6624 background/syren/
family_drivers rigor; 8d24dd7 plotting/diagnostics/designs edges; 2beb0f9
local-linear guard + doc contradictions). Full suite 813 OK after each.

One fix-wave item was investigated and NOT applied: a run_emulator guard
refusing anchor + trunk_epochs > 0 was drafted, then reverted because
test_training_pass_recipe exercises that combination deliberately (the
finding's "latent trap" premise was wrong for freeze_trunk=False); the
build_anchor docstring was corrected instead.

What remains open is exactly the Report-only section above; nothing there
may be fixed without a directive. The full reviewer reports live in the
session transcript, not in this repo.

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

Closed tickets are archived in [`backlog-closed.md`](backlog-closed.md),
grouped by subject. Nothing there is open work. To reopen one, move its
whole section back above `# Parked edge cases` in this file and add its
`- OPEN` index line; the reopen tooling requires the section to sit above
this heading.
