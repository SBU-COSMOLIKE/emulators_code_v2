# Execution backlog

Tracked in Git so unfinished fixes survive a new clone. Only the Architect
updates it. The daemon includes the Architect-sealed ticket update in the same
landing commit as the accepted fix.

## Contents

- [Open tickets](#open-tickets)
- [Parked edge cases](#parked-edge-cases)
- [Closed tickets](#closed-tickets) — a pointer; the sections themselves live
  in [`backlog-closed.md`](backlog-closed.md)

## How to read this backlog

Two readers: the Architect, who writes it, and the Red Team, who audits the
Architect and never edits. The Implementer never opens this file — its
directive carries the whole assignment — so the prose here stays compressed.
The saved SHA-256 catches an unexpected edit before the Architect writes again.

The watcher counts lines beginning exactly `- OPEN`, one per unfinished ticket
in the index below, each carrying `BUG FIX` or `NEW FUNCTIONALITY`. Never a
second `- OPEN` line inside a ticket.

Every ticket shows a **Red Team reopen count** (starts at zero, never resets)
and whether another reopening is allowed. Red Team review is advisory: the
Architect may accept and commit without it, and a `REOPEN` creates later work
rather than undoing the commit. On `REOPEN` the Architect restores the ticket
to Open, adds one to the count, then rules GO (repair) or NO-GO. NO-GO closes
it permanently and sets the reopening line to **barred by Architect NO-GO**; a
genuinely different defect then needs a new ticket. The sixth reopening makes a
ticket Low, so repeated disagreement cannot consume the queue.

A ticket already archived in [`backlog-closed.md`](backlog-closed.md) can still
reopen. The Red Team's handoff carries the full argument; the Architect moves
that section back here above `# Parked edge cases`, restores its `- OPEN` index
line, and expands the compressed entry from the handoff.

## Words used in open tickets

A **saved emulator** is the pair holding learned weights and the scientific
record needed to interpret them; the technical records call that pair an
**artifact**. A **saved-file format** or **schema** states which fields the
pair must contain. An **identity** is a saved fingerprint of the exact inputs,
settings, formulas, or files that produced an object; **provenance** is the
saved record of where data or weights came from. To **publish** a file is to
validate a complete temporary file and then place it at the final name a reader
uses.

An **adapter** is the Python bridge that hands Cobaya a result from a trained
emulator. A fine-tune **anchor** limits how far new weights may move from their
starting weights. A model's **domain** is the range of physical inputs on which
it may be used; **composition** is the formula that combines an emulator
correction with an analytic base calculation. A **resolved run record** stores
the settings the program actually used after defaults and automatic choices.

A **gate** is a named final check for a larger requirement; a **test** asks one
narrow question. CMB is the cosmic microwave background, MPS the matter power
spectrum, and PCE a polynomial chaos expansion. CUDA is NVIDIA's accelerator
platform, GPU an accelerator, MPI the message-passing system coordinating
several generator processes. CAMB is the upstream cosmology program, CosmoLike
the upstream program for several survey observables, and Syren an analytic
matter-power calculation vendored here.

# Open tickets

The Architect assigns priority at admission and records the reason.

- **Critical** — an Architect-only bug class: evidence that a current defect
  broadly breaks a central library workflow or systematically invalidates the
  library's scientific results. Never assigned to change which roles are
  active.
- **High** — a bug can make the science wrong, lose data, halt a core
  operation, or severely damage core behavior. State the demonstrated impact
  and why Medium is not sufficient. Urgency, a missing test, unfinished
  cleanup, or an expensive check is not by itself High, and the Red Team shows
  the same restraint. "The science can be wrong" is not enough alone: the
  defect must threaten a central scientific calculation, the training data, a
  served emulator result, or another primary library result. A defect confined
  to a plot, diagnostic ranking, optional report, or other supporting analysis
  stays Medium even when its output misleads — promote it only on evidence
  that it also corrupts a primary result or blocks a core workflow.
- **Medium** — a concrete problem reasonably likely during normal work, below
  the High boundary.
- **Low** — concrete but improbable edge cases.

Every ticket is also a Bug fix or New functionality. Severity sorts first, type
second: Critical bugs, High features, High bugs, Medium bugs, Medium features,
Low bugs, Low features. So a Low bug never jumps ahead of a Medium feature,
while a Medium bug precedes a Medium feature. Features are never Critical. The
words "after the backlog is closed" create a Low feature whose prerequisites
are every ticket already open when it was admitted.

A blocked ticket stays in its group with the blocker; the Architect may move to
the next permitted ticket while required hardware, data, an external decision,
or a named prerequisite is unavailable. New evidence may change a bug's
severity, with the reason recorded for every upgrade or downgrade.

A bounded repair may close an actionable bug once it removes the demonstrated
failure and leaves only a harmless exceptional case below Low, where complete
coverage would add disproportionate complexity. The exact remainder goes under
**Parked edge cases** with no claim of complete coverage. A parked
**LOW — EDGE CASE** has no `- OPEN` line, never enters a watcher count, and is
not a `--severity` choice; only a user request naming that exact ticket may
activate it as ordinary Low work. The class never hides a probable failure,
wrong primary science, data loss, or a broken core operation.

Backlog counts never change a role. Sol is the advisory Red Team when enabled
and does not implement tickets. Parallel work comes from the normal pipeline —
the Implementer codes a newly admitted ticket while the Architect audits a
previous commit and the Red Team reviews an earlier accepted one — and only
when the finite watch has another unused ticket slot. Each ticket still
consumes exactly one cycle.
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

**Severity: MEDIUM.** The anchor is deliberately unavailable in production,
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

- Partial on `main`: `2742156`. Complete local candidate: `25ac6d9` on
  `codex/unit24-anchor-hardware`, isolated CPU tests 299/299, plus the expanded
  identity gate, board self-test, and independent implementation review.
- Release stays `NO-GO` until the real GPU smoke and artifact readback pass.

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

- Landed and pushed as `03723c8`; local evidence 319/319 AI tests, all CPU
  child required results, five-gate dry run, board self-test, and independent
  GO.
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

- Landed and pushed as `18560d3`; CPU PASS, CUDA UNAVAILABLE with honest
  return code 2, plus schema and verdict controls, board self-test, and two
  independent reviews.
- Audit verdict: GO to land, NO-GO to close. Closure needs a CUDA workstation
  run with both assertion ids PASS and child return code 0; the two
  hard-coded-mode source mutations must fail there as proof the check bites.

</details>

<a id="open-workstation-debt"></a>
## Complete older cross-family workstation checks

### High-level summary

Several completed repairs still need real-workstation checks: rebuilding a
saved emulator, reading a real CMB calculation, and running an optimization
study with its persistent journal. Their CPU and source checks passed, but the
required CoCoA workstation is unavailable here, so an input, device, or
saved-file mismatch could stay hidden until a user starts a real training or
inference job.

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

- Carries the workstation debt formerly embedded in closed Units 13, 53,
  64/70, 96, and the AI-tree consolidation record. Does not duplicate the
  separately tracked Unit 24, DIDACTICS-62, or Unit 93 hardware runs.
- The old primary-worktree scratch-fixture drift is not open here; the later
  hygiene change that force-tracks only the disposable synthetic backlog
  fixture repaired it.

</details>

<a id="open-schema-v3-gate-fixtures"></a>
## Finish real workstation checks for the current saved-file format

### High-level summary

The repository's temporary saved-emulator examples now carry the scientific
records needed to test loading, reconstruction, and compatibility. The real
cs16 and cs8 CoCoA datasets predate that requirement, so their smoke checks
stop during setup and cannot yet exercise a real training, save, and prediction
run. Regenerating them is the evidence that the current file format works with
the configured scientific software and data, not only with small local
examples.

### Current status

**Ticket type: BUG FIX.**

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**OPEN.** Partial fix: commit `0fe2067` updated the repository fixtures and
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

- Fixture boundary: `ai/gates/checks/geo_paths.py`, `scalar_smoke.py`,
  `cmb_smoke.py`, `bsn_smoke.py`, `mps_smoke.py`; `0fe2067` completes it.
- The deployment manifest names the two required facts files beside the cs16
  and cs8 parameter tables. Both are absent in the current CoCoA checkout, so
  preflight correctly refuses instead of fabricating them.
- Required evidence: a CoCoA `geo-paths` run plus the four real workstation
  smoke gates. Never close from local CPU examples or a dry run.

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
`compute_data_vectors/generator_core.py`. Automatic retry must leave the closed
High failed-row safety repair unchanged.
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
Owners: dataset request/manifest, fixed facts, artifact compatibility, and
`emulator/syren_base.py`. A semantic formula change must change identity or
refuse; an unrelated documentation commit must not.
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

<a id="open-resolved-run-record"></a>
## Save every effective setting and reset each repeated study

### High-level summary

A resolved-run record should hold the settings the program actually used
after defaults and command-line choices were combined, so a training point can
be reproduced later. Several effective values are omitted, and repeated sweep
or tuning points can reuse an experiment after an earlier point changed its
weights, random state, or settings — so a reported result may be
irreproducible, and two sibling points may not start from the same fair
state.

### Current status

**Ticket type: BUG FIX.**

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**OPEN.** Resolved YAML and core study manifests exist; run totality, pristine
source identity, and root-configuration validation are incomplete.

**Severity: MEDIUM.** Reproducibility impact: a repeated study point cannot be
rebuilt from one authenticated source state.

### What is already fixed

Model, optimizer, several loss and schedule values, and composition facts are
saved in current results.

### What is missing

Persist effective rows/tails, update horizons, scheduler protocol, selection,
and pristine study identity. Rebuild each repeated point from one authenticated
source state and validate the complete configuration tree with close-match
errors.

<details><summary>Technical record for development tools</summary>
Owners: training resolver, experiment configuration, tune/sweep drivers, and
artifacts. Reordered points must stay identical; unknown or misnested config
and a state mutation must fail.
</details>

<a id="open-study-diagnostics"></a>
## Publish structured study and diagnostic results

### High-level summary

A study should distinguish a successful result from a failed or unavailable
point, and build tables and plots only from complete compatible values. Current
workers can turn failures into ordinary NaN-filled rows, and diagnostic and
plotting helpers accept empty, nonfinite, truncated, or mutually incompatible
inputs. The finished study can therefore look complete while hiding failed
work, or show a convincing comparison whose rows and curves are not the same
scientific cases.

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
Owners: sweep/tune workers, `emulator/diagnostics.py`, `results.py`,
`plotting.py`. Required witnesses: failure row, cleanup, empty/nonfinite,
truncation, log scale, color identity.
</details>

<a id="open-python-prose-review"></a>
## Make tracked explanations describe one coherent current library

### High-level summary

Tracked explanations should read as one deliberately designed current
library. Some READMEs, permanent notes, comments, docstrings, help text, and
diagnostics instead keep dated "hard user rule" labels, old bug-report names,
review waves, or later corrections sitting beside the earlier rule they
replaced, so a human has to work out which paragraph is the real rule.

Function explanations have a second problem: most functions in `ai/tools/` and
`ai/tests/` do not begin with a docstring saying what the function does, how,
what each input means, and what comes back. A short label or an old ticket name
leaves a student reverse-engineering the body.

Review the complete tracked repository after the existing backlog is closed,
rewriting policy chronology as current behavior plus its technical reason. Keep
a date only where it is part of the subject itself, such as a scientific
release or citation, and record why it must remain.

### Current status
**Ticket type: BUG FIX.**

**Red Team reopen count: 0.**

**Red Team reopening: allowed.**

**OPEN.** The function-docstring portion runs one file per commit, each proving
the change is docstring-only by comparing the two versions' syntax trees with
docstrings stripped, and each landing on a green full suite.

The bar is a strict mechanical census, adopted after a user spot check found
one-line docstrings surviving a "complete" claim: every callable with
parameters carries an `Arguments:` block, every value-returning callable a
`Returns:` block, no non-trivial body keeps a one-line docstring, and no term
of art or Python mechanic is left unstated.

Complete and census-zero: `emulator/` (40 files), `compute_data_vectors/`, the
five `cobaya_theory/` adapters, and `ai/tools/` (25 files). Also complete:
`documentation/emulator_code_guide.tex`, `emulator/README.md` Appendix D2, the
move of the 21 family drivers into `driver/`, the move of project instructions
to `.claude/CLAUDE.md`, and the removal of AI-workflow material from the main
README into `ai/README.md`.

Remaining: `ai/tests/`, and the repository-wide chronology rewrite outside
`emulator/`.

**Severity: LOW.** The user explicitly said "after the backlog is closed." This
improves maintainability and teaching but does not repair a current scientific
result, data-loss path, or halted core operation.
### What is already fixed

The permanent README and Python-style GO/NO-GO notes define the required
human-first voice. They require one coherent current-system account, complete
sentences, concrete examples for abstract ideas, accurate arguments and
returns, and explanations of non-obvious units, shapes, invariants, side
effects, and refusal reasons.

### What is missing
Build one complete inventory of tracked READMEs, the eleven permanent notes,
other tracked developer or scientific documentation, and every tracked `*.py`
file, then review module, class, method, and function docstrings, explanatory
comments, command help, diagnostics, and explanatory strings against it. A file
is not complete merely because it has a docstring. The prose must make sense to
a reader who has never seen the backlog, a Red Team report, an old ticket, or a
development-session label.

`ai/notes/python-changes-go-no-go.md` owns the required docstring shape and
`ai/notes/readme-go-no-go.md` the prose bar; this ticket does not restate them.
`ai/tools/` and `ai/tests/` are audited function by function against that
shape, and boilerplate copied between unrelated functions does not satisfy it.

Replace labels such as `DIDACTICS-62`, "Unit 8", `hard user rule`, wave or
round names, development dates, and ticket anchors with the behavior they stood
in for. Keep a date or historical identifier only when the subject becomes
false without it, and explain that necessity at first use.

Two hard boundaries. Git history is immutable: this ticket cleans current
tracked files and future commit-message templates only. And no computational or
scientific behavior changes — for a comment-or-docstring-only file, prove the
before-and-after syntax trees are identical after docstrings are removed; for
help, diagnostic, or other explanatory strings, require the executable diff to
contain only the intended literal changes plus focused exact-output and
return-code tests. Work in bounded non-overlapping batches against the one
inventory so no covered file is silently skipped.

<details><summary>Technical record for development tools</summary>

Mandatory example: `ai/gates/checks/d5_training_behaviors.py` must explain the
CPU calculations and training behaviors it protects instead of citing
`DIDACTICS-62`. The reference form for a non-trivial function is
`compute_batch_byte_terms`.

The review reports how many function and method definitions it inspected in
`ai/tools/` and `ai/tests/`, how many needed changes, and how many were left
without a docstring; the last number must be zero before this ticket closes.

Permanent-note findings return to the Architect — the Implementer and Red Team
never edit those eleven files. Priority dependency: every ticket open at
admission precedes this work. A behavioral defect found while reviewing prose
becomes its own bug ticket at its evidence-based severity, never a repair
inside this change.

</details>
<a id="open-emulator-audit-wave2"></a>
## Close the remaining verified emulator audit findings

### High-level summary

A user-ordered file-by-file audit covered all 40 `emulator/` files with ten
parallel reviewers; every finding below was reviewer-reported, and each
fix-wave item was Architect-re-verified against the code. Wave 1 landed as
ef2a85c. This ticket holds what was verified but NOT fixed.

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

**Severity: MEDIUM.** The remaining items are report-only findings in supporting
analysis; no primary result is known to be wrong.

**OPEN.** Narrowed to report-only plus deferred simplifications. Wave 1 landed
as ef2a85c; wave 2's fix candidates all landed across nine commits (ffb9aec,
6b126f1, 7adf66f, 5d17e31, a0406f0, e48c20c, 2bd6624, 8d24dd7, 2beb0f9), full
suite 813 OK after each.

One wave-2 item was investigated and NOT applied: a `run_emulator` guard
refusing anchor with `trunk_epochs > 0` was drafted, then reverted because
`test_training_pass_recipe` exercises that combination deliberately — the
finding's "latent trap" premise was wrong for `freeze_trunk=False`. The
`build_anchor` docstring was corrected instead.

What remains open is exactly the Report-only section above; nothing there may
be fixed without a directive. The full reviewer reports live in the session
transcript, not in this repo.

# Parked edge cases

- PARKED **LOW — EDGE CASE** **BUG FIX** — [Remove hidden covariance files left by forced process termination](#parked-cmb-covariance-cleanup)
- PARKED **LOW — EDGE CASE** **BUG FIX** — [Certify the vendored Syren formulas independently](#parked-syren-formula-certificate)
- PARKED **LOW — EDGE CASE** **BUG FIX** — [Guard the sizing probe if a stateful-forward family is added](#parked-memory-probe-stateful-forward)

<a id="parked-memory-probe-stateful-forward"></a>
## Guard the sizing probe if a stateful-forward family is added
### High-level summary

The batch-memory estimate runs one dummy forward on zeros through the live
model to measure autograd-saved activations. No current family changes state
during a forward pass, so the probe is harmless; a family with
batch-normalization running statistics or active dropout would make it mutate
model state or consume random numbers before training begins.

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
removes it after a normal success or handled failure. `SIGKILL` or another
uncatchable termination skips that cleanup, leaving one unreferenced hidden
file. Readers never use the private name and no partial final covariance is
exposed.

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
starting surface. A future edit to the vendored formula without retraining the
matching artifacts would combine the learned correction with a different
starting calculation and still produce finite values. No such drift is
demonstrated, and a formula registry or broad hash framework built only to
guard this hypothetical would make the scientific path harder to read — so the
case stays below Low.

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
