# Closed ticket archive

Closed tickets moved out of [`backlog.md`](backlog.md) so the Architect does
not load them on every turn. A closed ticket has no missing work of its own;
where later work is still required, the entry links the open ticket that owns
it.

Each entry is compressed to roughly 30 percent of the words it carried while
open: what was wrong, what fixed it, and the commit. Fuller evidence lives in
the commit itself.

Reopening still works from here, and the compression never has to preserve the
argument for it. The Red Team's handoff to the Architect carries the full
reason a closed ticket should reopen; the Architect moves the section back into
`backlog.md` above `# Parked edge cases`, adds its `- OPEN` index line, and
expands the compressed entry using that handoff.

<a id="closed-role-context-separation"></a>
## Give each AI role its own context limit

CRITICAL bug fix. One shared context setting meant a large Architect allowance
could exceed an Ollama Implementer's smaller model context and fail the
connection check before work began. `--architect-context`,
`--implementer-context`, and `--sol-context` now control only their named
roles; `--claude-context` survives as an Architect-only compatibility spelling.

<a id="closed-control-plane-live-state-compatibility"></a>
## Test a proposed controller against the saved state it must inherit

HIGH bug fix. A replacement mailbox controller was tested only against an empty
repository, so a new file format could be approved that failed on the first
real ticket or recovery record, halting the controller after landing. The
protected check now starts from a disposable copy of the live records, which
the proposed controller must read in a fresh process while preserving their
workflow meaning. A state-schema change requires an exact migration
declaration. Coverage: `ai/tests/test_protected_control_plane_shadow.py`.

<a id="open-syren-amplitude-aliases"></a>
## Refuse conflicting amplitude names before calculating Syren matter power

HIGH bug fix. The Syren formulas accept the primordial amplitude as either `As`
or `As_1e9 = 10^9 As`, and a Cobaya run can supply both. The helper took
`As_1e9` without checking agreement, so a saved network could read one
amplitude while its analytic Syren starting surface used another: both finite,
together describing no single cosmology, and shifting the analytic linear-power
baseline by about 77 percent. `syren_params_from` in `emulator/syren_base.py`
now compares `As_1e9` against `1e9 * As` under one tolerance sized to float32
storage and raises a `ValueError` naming both values. Both production call
sites pass through that function before either analytic formula, so the
generator refuses before writing a row and the adapter before any `Pk_grid` or
derived state exists. Candidate `f30f406`.

**Red Team reopen count: 1. Reopening: barred by Architect NO-GO.** The Red
Team correctly observed that `emul_mps.calculate` calls the linear learned
predictor once before `syren_params_from` raises, so an earlier "before any
learned predictor" wording overstated the boundary (corrected above). The
durable authority (`saved-emulators.md`, "Syren parameter
aliases must agree") requires refusal before either analytic formula and no
surviving derived state, and the landing meets both. The finding is Low, below
this ticket's High severity, and causes no wrong science, data loss, or halt.

<a id="closed-syren-amplitude-aliases"></a>

## Documentation and teaching

### Make backlog tickets and the gate guide readable

The backlog mixed open and closed work in one list and used internal labels
such as "unit 8" that did not say what was wrong. It now has a linked Open
section and a grouped Closed section; the gate guide gives real commands and
visible results and explains the runner's setup, logs, states, and options.
Landed as `b906147`.

### Remove the obsolete README trimming quota

A former ticket required a fixed 15 percent cut in README words even when the
words were useful. Detailed teaching now lives in folder guides, so trimming is
judged by clarity. Retired by user directive; no implementation commit.

### Explain every developer test

The test guide listed filenames with labels that did not say what input was
used or why a refusal mattered. Every immediate Python file in `ai/tests/` now
carries a concrete example, action, accepted result, refusal, and scientific
reason, and an automated check rejects a stale inventory row. Landed as
`a875f3f`, with the scientific-explanation pass in `9a55c7b`.

### Make the YAML workflow diagram readable on phones

The first `example_yamls` diagram was too wide for a phone. It is now a
five-step vertical path (choose, copy, edit, check, run) with the same
sequence stated in prose. Landed as `ac3b3eb`.

### Move detailed appendices out of the main README

The README made a new user cross pages of specialist material before the first
run. It now keeps a five-step startup path and sends YAML, data generation,
Cobaya, emulator, Syren, and AI detail to the folder guides; the root README
went from 3,717 to 1,122 lines. Landed as `b0aa890`.

### Make the permanent notes durable and Python style mandatory

The permanent notes read like a dated development diary and Python readability
was treated as a preference. The eleven notes now record neutral current rules,
only the Architect may edit them, and `python-changes-go-no-go.md` makes
readable Python a GO/NO-GO condition.

### Correct CoCoA setup and project paths in the guides

Four guides duplicated CoCoA setup and used an invented
`projects/lsst_y1/cobaya` folder. They now point to the official instructions,
place editable YAML directly under `projects/lsst_y1`, and separate user-copied
YAML from generated data. Landed as `b87b9f7`.

### Add a beginner guide for the AI tools

Users had to read the large AI guide to find which tool to run and whether it
changed files. `ai/tools/README.md` now covers all five programs, daily
commands, visible results, stopping, recovery, and bundle transfer, while
`ai/README.md` keeps the first-ticket path. Landed as `2086207`.

### Make all AI README workflows vertical

Six workflow diagrams were hard to follow on narrow screens. They now read top
to bottom using visible actions, with the prose beside each picture stating the
same sequence, so no safety rule depends on interpreting the graph.

### Add a beginner guide for Cobaya adapters

The `cobaya_theory` folder had no direct path from a saved emulator to one
checked Cobaya evaluation. Its guide covers the five adapters, setup versus
evaluation, file matching, device behavior, physical limits, and the NumPy 1.x
boundary.

### Add a beginner guide for example YAML files

The ten shipped YAML files had no guide for choosing a starting point or for
separating a syntax check from scientific validation. The new guide shows how
to choose, copy, edit, check, and run one template, and states that a passing
parser check does not prove the scientific settings are right.

### Simplify the first AI workflow picture

The first diagram introduced ten boxes before a reader understood the process.
It now shows only the user request, Architect plan, Implementer work and tests,
and Architect review; Red Team, repair loops, and worktrees come later.

### Add a beginner guide for generating training data

The generator programs build the tables used for training and validation, but
the folder had no guide to their outputs, failure flags, seeds, memory, MPI,
resume, or append behavior. The new guide gives a six-step first path and moves
family detail to question-led appendices.

### Explain why the three-role system saves scarce AI tokens

The AI guide used roles without explaining why a student would accept the extra
structure. It now says unlimited access may make the system unnecessary and
explains reserving expensive reasoning for the Architect and optional Red Team
while a simpler model writes and tests code.

### Rewrite AI appendices in beginner language

The appendices used lane, dispatch, worktree, and schema before a reader could
connect them to a file or action. FAQs A-H now define or replace those terms
where they first appear and explain stopping, role folders, Red Team scope,
recovery, and archive transfer.

### Protect README and Python explanations with a GO/NO-GO contract

README prose and explanatory Python text could become factually polished but
unusable for a physics student. The eleventh permanent note requires concrete
examples, local definitions, and exact evidence, and passes an editorial review
against private standards; the SHA guard prevents accidental note drift.

### Remove the overlapping mailbox-diagram label

A self-loop label in the mailbox lifecycle picture rendered on top of another
label. The loop was removed and the unchanged failure meaning written beside
the work-in-progress box.

### Rename `texnotes` to `documentation`

Teaching sources and the activation-function notebook sat under an unclear
folder name. They now live under `documentation/`, with links, build paths, and
custody text following.

### Make the AI guide role-first and visual

The guide mixed role rules, command details, and internal mechanics before
saying who decides and who changes code. It now opens with stable role
boundaries, selectable Claude models, Architect-only GO/NO-GO, bounded Red Team
scope, and visual workflows.

### Render README equations correctly

Four formulas appeared as raw bracketed text. Both delta-chi-square equations
and the default activation equation now use GitHub-compatible display-math
fences; only the delimiters changed. Landed as `66f7046`.

### Reorganize the root and AI READMEs around short startup paths

Both guides mixed first-use instructions with reference material. They now
separate a short main path from question-led appendices grouped by subject.
Landed as `eb17489`; the later `c91791a` repaired the disposable
primary-worktree fixture this ticket left behind.

<a id="closed-ollama-documentation-model"></a>
## Use GLM-5.2 Cloud in the Ollama examples

LOW documentation change. The guides used Qwen as their example Ollama
Implementer; they now use `glm-5.2:cloud`, noting that it needs an Ollama
account and runs prompts through Ollama's cloud service. Runtime selection
stays explicit through `--implementer-provider ollama --implementer-model
MODEL`; no model name is compiled into the daemon.

<a id="closed-ollama-ping-visible-thinking"></a>
## Let reasoning-capable Ollama models pass the connection check

HIGH bug fix: the documented Ollama model could not pass the check required
before an unattended watch. The probe demanded that the model's entire visible
answer be the requested marker, and GLM-5.2 Cloud printed its reasoning first.
The no-work probe now passes `--hidethinking` while still requiring an exact,
unpredictable reply.

## AI roles, user controls, and handoffs

### Recover a completed Implementer return after validation refuses it

HIGH bug fix. Harmless Markdown list formatting made the evidence validator
reject a completed candidate, and the watcher then idled instead of preserving
the work. A restart now revalidates the saved return and sends it to the
Architect without rerunning the Implementer, and the parser accepts a required
heading inside a list.

### Restart cheap role work without rebuilding the Architect plan

HIGH bug fix: the old manual recovery could halt normal ticket operation and
invited unsafe file moves. Ctrl-C can leave a request outside the waiting queue
with unfinished work; `--restart-implementer` and `--restart-redteam` requeue
one exact handoff, reusing the Architect's saved plan and refusing ambiguity or
a completed result.

### Explain every Architect candidate review in the terminal

MEDIUM new functionality. A formal `GO` or `NO-GO` says whether work advances
but not how close the Implementer came. Each candidate audit now ends with a
seven-line assessment naming strengths, remaining work, file scope, and the
next action, in five plain categories that avoid false precision.

<a id="closed-subagent-discretion"></a>
### Let the Architect decide when helpers add real value

Requiring a helper for every ticket could spend more credits coordinating a
five-line correction than performing it. Every directive now carries exactly
one visible choice: `Subagents required` with bounded named jobs, or `Subagents
not required` with a concrete Architect-authored reason. The validator refuses
empty, vague, contradictory, or Implementer-authored waivers. Landed as
`0ff77fa`.

<a id="closed-structured-role-contract"></a>
### Protect one machine-readable role contract

Role permissions, timing limits, candidate identity, landing authority, backlog
ownership, and the single-review rule were repeated across Python and prose, so
an edit could change one copy without making the contradiction obvious. A
protected JSON-compatible YAML file is now the machine source of truth, and an
Implementer candidate cannot change the contract, its reader, the eleven notes,
their guard, or protected role files. Landed as `8611e1e`.

### Prohibit new monkey patches without forcing a wholesale rewrite

A monkey patch changes executable behavior while Python runs, so an apparently
local test can change a later one. The permanent Python contract and all three
role contracts now reject a monkey patch that is added, copied, retargeted, or
broadened, while naming the ordinary local fakes that stay allowed. Two
existing sites are recorded in the High queue.

### Use a 4,000-character warning for one bug repair

The earlier size warning was close to a few clear Python lines and made
ordinary bounded repairs look disproportionate too early. The Architect now
becomes strongly suspicious only above 4,000 added-plus-deleted characters
outside `ai/tests/` and `ai/gates/`. It stays a warning, not a limit.

### Make the Architect the only user-facing role

Direct messages to the Implementer or Red Team bypassed the role that owns
scope and final decisions. Public send and ping commands now accept only
Architect requests, and a carried handoff must remain unchanged.

### Filter Red Team discoveries by severity

Discovery runs could create tickets for edge cases the user did not want to
pursue. `--severity` selects severe failures only, probable normal-use bugs, or
every concrete finding, defaulting to `medium`; the Architect may accept,
upgrade, or downgrade a finding with a reason.

### Limit changed characters without allowing unreadable code

Maintenance tickets needed a way to reject changes that touch too much code.
`--max` limits added plus deleted characters; zero means unlimited, and an
unmeasurable or conflicting candidate refuses. The Architect must still reject
shortened names, collapsed logic, or removed explanations. Landed as `859dab2`.

### Give Sol its own saved worktree

Sol started in the repository folder reserved for the user. It now creates and
reuses an independent `mailbox-sol` worktree, with repository, path, branch,
role, tool, and notes identity checked before and after each launch. Landed as
`1e17fe2`.

### Create and reuse Claude's primary coordination worktree

Claude sessions guessed at a work folder or used the user's main checkout. The
first live use now creates or deliberately adopts one worktree, and later runs
validate and reuse its exact repository, path, and branch. Fixture drift was
repaired by `c91791a`.

**Remaining:** a user who owns the preserved `amazing-keller-e798b6` transport
must deliberately migrate or adopt it; the tool will not mutate that user-owned
state automatically.

### Package unfinished backlog work for another developer

A user out of credits may need to send one snapshot of unfinished work onward.
`backlog_bundle.py` creates a `.tar.xz` package, checks it without writing, and
imports only into a new ignored review folder. The archive binds repository and
base identity, sizes, SHA-256 values, paths, and exact bytes.

### Allow a two-role run without Red Team

Some work needs only an Architect and Implementer, but the watcher always
assumed a Red Team route. `--skip-redteam` and `--no-red-team` disable it while
preserving its waiting messages for a later three-role run; demand and
cycle-zero count only enabled routes.

### Let the user choose Claude models by role

Architect and Implementer were tied to expensive default model names.
`--architect-model` and `--implementer-model` accept aliases or full Claude
model IDs while the mailbox routes keep their stable role meaning; invalid
names refuse before launch.

### Require detailed plans that simpler Implementers can follow

Earlier handoffs assumed an expensive Implementer would fill in missing design
choices. Architect and Red Team directives must now name exact files,
algorithms, invariants, failures, tests, expected results, exclusions, and stop
conditions; incomplete or choice-leaving packets refuse. Landed as `866b30b`.

<a id="closed-failure-catalog-consistency"></a>
## Keep the Implementer failure catalog synchronized with its controls

LOW bug fix. One catalog entry repeated the current 90-minute setting as a
literal value, and its code references could go stale after a rename. The
`timed_complexity` entry now names
`role-contract.yaml::runtime.implementer_review_minutes` instead of its value,
and the role-contract tests confirm unique IDs and use Python's syntax tree to
find every referenced function or class in its named file.

## Scientific code, data handling, and gates

<a id="open-mps-serving-domain"></a>
<a id="closed-mps-serving-domain"></a>
### Reject matter-power requests outside calibrated ranges

Matter-power serving accepted axes and surfaces it could not support. Saved and
direct interpolation axes must now be one-dimensional, finite, strictly
increasing, and long enough for the spline, with positive wavenumbers and
exactly matching finite surfaces; a redshift outside the saved interval always
stops. A wavenumber tail is allowed only through the logarithmic boundary
continuation, a visible option on by default. Landed as `a0633ad`.

**Remaining:** earlier configuration errors are
[Validate matter-power requests before a run starts](#open-mps-request-contract);
a future Syren source edit keeping the old law name is parked under
[Certify the vendored Syren formulas independently](backlog.md#parked-syren-formula-certificate).

<a id="closed-background-protocol"></a>
### Reject invalid redshift grids, coordinate pairs, and nonflat cosmologies

Plausible-looking distances could be served for a calculation this
implementation does not support. Background distances now start from an ordered
Hubble grid anchored at redshift zero: nonfinite, negative, duplicate,
reversed, or unanchored grids refuse at training, rebuild, and inside the
integrator, and the Cobaya bridge keeps each two-redshift request as an exact
ordered `(N, 2)` pair, and generation and serving refuse a nonzero `omk`.
Landed as `a3b345e`. Curved-distance formulas remain user responsibility.

<a id="closed-cmb-serving-contract"></a>
### Reject physically impossible CMB spectra before serving them

The CMB bridge published results without checking them. It now validates the
complete local result first: TT, EE, and PP must be finite and nonnegative, and
TE stays signed but cannot exceed `sqrt(TT) * sqrt(EE)` where all three share a
stored multipole. An invalid prediction leaves the caller's state unchanged.
Landed as `2016c40`. Unit and multipole-factor conversion is still deliberately
refused pending its own ticket.

<a id="closed-compatibility-manifest-removal"></a>
### Remove the duplicate compatibility manifest from saved emulators

Each saved emulator already records the model recipe, geometry state, analytic
law, and composition mode needed to rebuild it. A second manifest copied those
facts and added fixed labels such as `model:...:v1` that inspected nothing. It
was removed as one change (431 lines deleted, 10 added), since splitting it
would leave a partly removed file format. Landed as `8030857`; this
deliberately breaks the brief alpha format, so an artifact written while the
manifest existed may need regeneration.

<a id="closed-model-recipe-simplification"></a>
### Keep only the model recipe checks needed to rebuild an emulator

A saved recipe stops a later software default from silently rebuilding a
different network, but the old implementation also repeated numerical rules the
constructors already check. The reader now checks the complete closed
rebuilding description before importing model code, and numerical rules stay
with the constructor that uses each value. Landed as `346e65b`, which also
renames the module to `model_recipe.py`.

<a id="closed-transfer-state-digest-simplification"></a>
### Remove duplicate hashes for embedded transfer-model weights

A transfer emulator stores its base-model tensors in the HDF5 file and loads
them strictly, but the old path also hashed those tensors repeatedly into
attributes and nested configuration records. Commit `64fa00a` removes the
digest declarations, structure walks, repeated comparisons, and the digest
helper (50 characters added, 13,072 deleted as one format cleanup). Missing,
extra, or wrong-shaped tensors still fail strict loading.

<a id="closed-training-history-load-simplification"></a>
### Stop revalidating training history while loading an emulator

Training histories record how a run progressed; they do not define the network
used for prediction. The old rebuild path nevertheless parsed a large
optimizer, schedule, pass-order, and history grammar before loading a model.
Commit `a4f8fa8` removes that duplicate parser from `results.py`: the writer
still requires five finite, compatibly shaped history arrays, and rebuild keeps
the training mapping as opaque provenance bound into output identity.

<a id="closed-dark-energy-coordinates"></a>
### Preserve time-varying dark energy from data generation through serving

Cobaya may sample `w0pwa = w + wa` and calculate `wa` before a theory component
runs. The old path could overlook that value and silently substitute `wa = 0`,
producing a smooth finite spectrum for constant dark energy even though the
user asked for a time-varying cosmology. Generation and serving now share one
checked conversion: direct `w`/`wa` and transformed `w`/`w0pwa` resolve to the
same pair, repeated forms must agree under one tolerance, and zero evolution is
assumed only under an explicit constant-`w` or cosmological-constant law.
Landed as `32328be`, with the permanent-note rule in `8b7f991`.

<a id="closed-artifact-recipe-totality"></a>
### Save every model-building setting needed to rebuild a trained model

A saved emulator must rebuild the exact model that was trained, never guessing
from current Python defaults or accepting a plausible recipe describing
different activation curves, layer counts, geometry, training phases, analytic
formulas, or transfer-base weights. The writer now compares the saved
description against the live model and records the executed training plan and
transfer state; the reader validates both before constructing model components.
Landed as `dd44234`.

<a id="closed-generator-ingress"></a>
### Validate generator inputs before creating output files

HIGH: malformed but normal-looking input could create an undersized or
differently defined dataset that later training treated as valid science. A bad
parameter order, covariance, fiducial, grid, prior bound, or command setting
was discovered only after output work began, and MCMC rows distinct in memory
could collapse to duplicates once saved as `float32`. The generator now
validates the whole request before creating output and counts distinct rows at
the precision readers actually receive, refusing before a draft exists when
that count is too small. Landed as `9d53a51`.

<a id="open-artifact-output-identity"></a>
<a id="closed-artifact-output-identity"></a>
### Give scientifically different emulator files different names

Two runs representing different spectra, quantities, selected rows, loss modes,
or source models could share one output name, so a later run could replace a
valid earlier result. A saved name now starts with the output family and
product and ends with a 32-character digest binding the resolved model and
training settings, executed rescaling mode, published generations and row
order, composition rule, and reused source pair. Rebuild checks the identity
and the exact `.emul` weight digest before PyTorch loads anything. Landed as
`fa1ec12`.

<a id="open-padded-head-identity"></a>
<a id="closed-padded-head-identity"></a>
### Stop artificial padded values from mixing with physical bins

CNN and Transformer heads use artificial positions so physical groups of
different lengths share one rectangular tensor. The old saved representation
could not prove which positions were physical, and some operations could turn
an artificial zero into a nonzero value that reached a real prediction. The
repaired heads save the physical-position map and validity mask, reapply the
mask after every operation that could revive a padded position, and gather only
recorded physical positions. Landed as `32f5b48`.

<a id="open-active-model-validation"></a>
<a id="closed-active-model-validation"></a>
### Reject invalid model settings before building the model

Model settings are now checked before the program opens training files, selects
an accelerator, or creates layers, so wrong types, impossible sizes, nonfinite
numbers, and unsafe output activations stop with a message naming the exact
setting to correct. The same rules repeat inside the public constructors, so a
caller building a model directly cannot bypass them. Landed as `08172db`.

**Remaining:** the gradient problem at the exact origin is
[Preserve the power activation gradient at zero](#open-power-zero-gradient).

### Calculate sigma-eight at the conventional physical radius

The matter-power bridge used a literal 8-Mpc radius although its saved
wavenumbers use inverse megaparsecs, so for `h = 0.64` it could report a value
near one as sigma-eight when the conventional answer is near 0.64. It now uses
`R = 8/h` Mpc, requires the exact stored redshift zero, and refuses a
wavenumber grid whose tails or resolution cannot support the integral; the CAMB
reference agrees within the declared 0.2 percent. Landed as `3134cd5`, with
permanent-note commit `ee43ec0`.

<a id="open-pce-strictness"></a>
<a id="closed-pce-strictness"></a>
### Stop the polynomial emulator from saving a fit that failed its accuracy limit

The polynomial emulator kept its first output pattern even after every pattern
missed the accuracy limit, so it could save a base its own leave-one-out check
had rejected. The fit now judges input bounds, coefficients, and the complete
matrix in the same 32-bit format used after saving; a failing pattern is
removed and the smaller matrix rechecked, equality with `loo_max` fails, and no
emulator is created when no pattern remains. Landed as `dd07caa` with
permanent-note commit `aaac2d7`.

**Remaining:**
[Refuse polynomial-emulator requests outside the fitted parameter range](backlog.md#open-pce-domain-enforcement).

### Refuse invalid values at every public prediction boundary

A saved emulator could ignore the output transformation recorded during
training, and prediction trusted intermediate arrays, so a Boolean, nonfinite
number, wrong width, or broadcastable matter-power row could reach a likelihood
as apparently valid science. The reader now serves only artifacts that record
the supported untransformed target, and every intermediate is checked before
publication. No adapter leaves a partial sampled-point result after a later
calculation fails. Landed as `6c21155`.

### Keep generated datasets complete through training

A generator wrote related files at different moments, so resume or append work
could combine rows, payloads, axes, or failure flags from different results,
and training opened flat filenames without proving one generation. Work now
stays private until one complete read-only generation is selected, each YAML
parameter filename finds one authenticated generation, and a compare-and-swap
stops stale resume work replacing a newer result. Landed as `fa8f170`.

**Remaining:** exact append, recovery of an interrupted private draft,
persisted sampler state, and old-generation cleanup are
[Continue generated datasets exactly and manage old generations](#open-dataset-continuation-features).

### Keep failed physics rows out of training datasets

A failed physics calculation left a finite zero vector of the expected shape,
so training could mistake the placeholder for a real cosmology. A generation
can no longer publish while any row is marked failed, and staging requires the
authenticated mask and removes failed rows before cuts, seeded selection, and
pool-size counting. Landed as `fa8f170`.

**Remaining:**
[Retry failed generator rows reproducibly](backlog.md#open-generator-failure-retry).

### Stop training before it can save an unreadable emulator

Training could finish without the scientific record prediction requires, then
save a file the same library immediately refused to reopen. It now validates
the training and validation facts before choosing a device, opening a
warm-start or transfer artifact, or constructing the experiment; saving writes
schema 3 only and refuses invalid input before changing output files. Landed as
`0fe2067` with permanent rule `b6c7afd`. Arbitrary constructor coverage closes
under [the saved-recipe ticket](#closed-artifact-recipe-totality).

**Remaining:** the real-workstation
[gate-fixture checks](backlog.md#open-schema-v3-gate-fixtures).

### Publish and load each saved emulator as one authenticated pair

A saved emulator's weights file and scientific-record file shared no
fingerprint, so a crash or file swap could join plausible but unrelated files,
and loading did not restrict PyTorch to tensor data. Both are now staged and
checked before their final names change, the record stores a shared identifier
and the exact SHA-256 of the weights, and rebuild refuses a swap before model
construction using a tensor-only load. Landed as `9711160`.

### Authenticate fixed facts in the artifact and adapter chain — Unit 84

Saved fixed scientific settings had to stay consistent from the training
artifact through to the Cobaya adapter. Unit 84 supplied the artifact half:
both sides use the audited fixed-facts path instead of rebuilding those values
independently. Committed with Unit 85 as `d3b9289`. The joint audit re-ran
every identity gate, proved the verbatim-move claim at the AST level, and ran
an independent mutation probe: disabling the predict-path support law reddened
two legs in each of five gates, so the gates are production-coupled.

**Remaining:** [the gate-fixture ticket](backlog.md#open-schema-v3-gate-fixtures).

### Authenticate fixed facts in the artifact and adapter chain — Unit 85

The adapter half of the same change, sharing Unit 84's code version, commit
`d3b9289`, and audit record.

### Save and verify an artifact's composition mode — Unit 96

A reader could infer plain, neural-PCE, or transfer behavior from whichever
HDF5 groups happened to exist. Schema-v3 artifacts now declare the native
composition mode and refined state, and the reader checks that declaration
against the exact payload and resolved YAML before loading weights. Landed as
`3d47318`; 30 forged rows refused.

**Remaining:** real-dump confirmation in
[the workstation ticket](backlog.md#open-workstation-debt), the schema-v3 smoke
fixture in [the gate-fixture ticket](backlog.md#open-schema-v3-gate-fixtures).

### Authenticate the Grid2D constant mask — Unit 96 rider

A saved Grid2D model could carry a constant mask with nothing proving the
ordered mask unchanged. Saves now record its SHA-256 for the main geometry and
transfer base and rebuild checks it before creating the model; missing
declarations, changed order, and mask data on non-Grid2D artifacts refuse.

### Enforce boundary and interior support policy — Unit 94

Generated parameter samples needed one explicit rule for points near the
allowed boundary and points in the interior. The old candidate was ported to
current main and rechecked as `f046085`. Do not merge the obsolete `a0a03a9`
branch again.

### Validate each generated row before marking it successful — Unit 56

A row could be marked successful before the serial, MPI, resumed, dtype, shape,
finiteness, and byte-readback checks agreed. Every path now uses one predicate
and clears the failure flag only after the written bytes are read back exactly.
Landed as `e885a8d`.

### Centralize background quantity and unit validation — Production Unit 62

Background quantity and unit pairs were checked in several places and could
disagree. One registry now controls configuration, geometry, rebuild, and the
Cobaya background adapter, so wrong pairs and non-string, nonfinite, Boolean,
or quoted offsets refuse before save or inference. Landed as `c6fca01`.

### Require CPU-normalized saved model state — Units 64 and 70

Saved `.emul` state needed direct proof it holds a nonempty tensor dictionary
with every tensor on the CPU. A ninth independent result checks those bytes
without a load-time device override. Landed as `fb5302e`.

**Remaining:** the CUDA, CosmoLike, deployment-dump, and `.cpu()` removal run
is in [the workstation ticket](backlog.md#open-workstation-debt).

### Preserve Grid2D row identity during staging — 25M-32/33

Grid2D staging could lose the generator's seeded row order while moving through
raw, base, parameter, data-vector, and index arrays. Resident and memory-mapped
paths now preserve one exact row identity and check all row counts before
allocating transformed targets. Landed as `c688489`.

### Authenticate optimization-study identity — Unit 53 repair

An optimization study could reuse results created with different scientific
inputs, family choices, or implementation rules. One manifest now fixes that
identity before workers start and prevents failed or stale trials from becoming
the winner.

**Remaining:** the real Optuna journal run is in
[the workstation ticket](backlog.md#open-workstation-debt).

### Repair the generator-ranges gate

The range gate could miss an old header format because GetDist might accept or
reject comment rows before the intended assertion ran. `generator_ranges.py`
now checks the producer-owned rows-only sidecar directly, asserting exactly one
three-token bounds row per sampled parameter in order, so the retired
`# weights lnp ...` header is rejected even on GetDist 1.6.2, which accepts
that comment.

### Repair the cross-family transfer-refusal gate

The cross-family transfer check failed early because its fixture omitted
ordinary required data, masking the rule it was meant to test.
`transfer_identity.py` now gives its Grid2D fixture valid `n_train` and `n_val`
so the candidate reaches the family-kind check, and the refusal must name
`a transfer never crosses families`.

### Resolve parameter names for numbered chain files

A file such as `chain.1.txt` could fail to find the shared `chain.paramnames`
declaration. One resolver now applies the numbered-root fallback in ordinary,
fixed-facts, and scalar staging.

### Retire a stale rebase ticket — Unit 90

Unit 90 looked unfinished even though its accepted implementation was already
part of main. The entry was reconciled against Git history rather than merging
the work again: commit `50f1c63` is an ancestor of main through `ce99f87`.

### Validate the CMB covariance package — Unit 13

CMB covariance generation needed explicit wiring and independent failure checks
for each validator. The accepted package combines the scientific calculation
with the command path that runs it. Substance `2fd8a9d` plus wiring `7583019`;
durable owner `ai/notes/families-scalar-cmb.md`.

**Remaining:** Torch CMB identity and real-CAMB byte identity are in
[the workstation ticket](backlog.md#open-workstation-debt).

### Recheck sampled parameter order before loading artifact weights

A forged schema-v3 record and matching sidecar could agree with each other
while disagreeing with the rebuilt input geometry. `rebuild_emulator` now calls
`fixed_facts.check_names_match` immediately after rebuilding that geometry and
before output geometry, PCE or transfer reconstruction, model construction, or
weight loading, so a coordinated rewrite to reverse parameter order is refused
with `torch.load` never called.

### Use an Ollama model as the Implementer

LOW new functionality. Both role commands were built around Claude, so a user
who could run a capable open-weight coding model locally still spent Claude
allowance on token-heavy implementation. The providers are now independent:
`--implementer-provider ollama` routes the Implementer to a named Ollama model
through its headless coding integration, with the same isolated worktree,
checkpoint hook, directive, and evidence requirements. The default remains
Claude.

**Remaining:** the workstation running the watch must install Ollama, start its
service, and download a coding model with enough context for the ticket.

## Mailbox and watcher behavior

### Start every role turn with an empty provider conversation

The watcher already started a separate provider conversation per mailbox turn,
but the launch commands did not forbid saving those sessions, so the
fresh-context boundary depended on nobody later adding a resume option. Claude
dispatches are now explicitly non-persistent (`--no-session-persistence`) and
Sol dispatches ephemeral (`--ephemeral`). Landed as `5b3f84f`.

**Remaining:** one role turn can still compact if that turn alone reaches its
context limit; raising the limit or splitting the ticket addresses that case.

### Stop cleanly when an AI account runs out of tokens

Claude or Sol can exhaust an allowance mid-watch, and the daemon saved the
request but kept polling without telling the user. A verified account-limit
message now stops the watch with `Error: <role> is out of tokens`. The exact
request is preserved in `failed/`, a role that already started may finish, and
no later request starts in that pass. Transient rate limits and context-size
failures are not mislabeled as exhaustion.

**Remaining:** retrying is a user action after credits are restored, because
automatically resetting a partially edited Implementer worktree could lose work.

<a id="closed-user-main-primary-sync"></a>
### Let a clean user update on main reach every AI worktree

An ordinary commit or pull in the user's clean main folder left the three saved
AI worktrees one commit behind, and the next watcher command refused to start
because a user commit carries no ticket-landing receipt. That folder's exact
commit is now recognized and every clean idle AI worktree advances to it, but only
when no ticket or candidate is active and files and index are clean. Landed as
`35f39b4`.

<a id="closed-clean-all-ai-worktrees"></a>
### Remove every AI-created worktree and branch on explicit request

Old sessions can leave enough worktrees, branches, and mailbox history that the
daemon cannot safely decide which folder is current. `--clean-all` discards all
local AI work, including dirty files and unmerged commits, and runs before
primary-folder selection so it stays usable when `--once` is ambiguous. It
leaves remote records, tags, stashes, and non-AI Git work intact and never runs
automatically. Landed as `957afc4`.

<a id="closed-provider-connectivity-ping"></a>
### Check whether Claude and Sol can answer before starting work

The old ping only placed a message in the mailbox; it could not say whether
either service was logged in and able to answer. Bare `--ping` now makes one
small live request to each, and each must return the exact nonce-bearing reply
within two minutes. No worktrees, mailbox messages, backlog changes, or ticket
cycles are created. Landed as `24e7888`.

<a id="closed-role-contract-coverage"></a>
### Make the protected role contract cover every stable security authority

The protected YAML named only part of the authority the tools use, so a later
Python edit could weaken a guard, change a saved role worktree, or admit a
control file without the contract visibly changing. Schema 2 records the eleven
notes, three role files, both guard files, nine trusted tools, the Architect
backlog, candidate-forbidden paths, size limits, and the three role worktrees,
and the daemon compares them against what it actually uses. Landed as
`96766d6` as one atomic migration, since splitting it would leave two
disagreeing sources of authority.

### Require one adversarial review of protected policy changes

The role files and eleven permanent notes control how later work is planned.
With Red Team enabled, the Architect shows it the exact proposed wording once,
so an unnecessary, oversized, or contradictory rule change gets an independent
challenge. Red Team gives one read-only advisory response and the Architect
decides: no revision loop, second review, veto, or post-landing review.

### Pause long Implementer work for an Architect complexity review

An Implementer could spend hours expanding one repair before the Architect saw
the approach had become too complicated. The watcher now asks it to pause after
90 minutes, save coherent progress, and explain size, remaining work, elapsed
time, and complexity. The pause is a progress review inside the same ticket and
cycle, so it cannot be accepted as a candidate or landed.

### Require a checkpoint decision before implementation resumes

A pause could return to the Implementer without an explicit Architect decision,
and a checkpoint with no new commit left no immutable candidate to inspect.
Every timed checkpoint now needs a new clean commit and one fresh same-cycle,
same-mode GO or NO-GO before work resumes; a checkpoint cannot send landing GO.
Landed as `4e21b6f` under a recorded Critical size exception, because prompt
authority, checkpoint identity, decision binding, and atomic output refusal had
to close together or the pause would stay bypassable.

### Keep GitHub commit messages readable

Landing commits replaced the Implementer's explanation with an internal ticket
label, so GitHub history could not say what changed, why, or which checks
passed. Every AI-authored commit now needs a concrete subject and a short
Markdown body giving the observed problem, the change and its boundary, and
exact evidence. Implementation `861acd5`; permanent rule `3e22a1b`.

### Keep mailbox dry runs read-only

A dry run could move a malformed or placeholder message even though the user
asked only to preview. Dry-run mode now performs the same validation without
claiming, moving, or writing any message state; only a real dispatch claims a
pending file into the work-in-progress folder.

### Use readable sentence-case terminal output

Daemon output used dense separators and all-capital phrasing that was hard to
scan. User-facing lines now use sentence case and semicolons while preserving
exact protocol and acronym text.

### Do not require a reply to an explicit terminal message

Every inbound message demanded a reply, including one that said the
conversation was finished. An exact terminal or no-reply message is now exempt
while any ambiguity still requires an outbound response.

### Keep claimed mailbox work current and publish outcomes safely

A claimed message could become ambiguous when newer work arrived, a timeout
occurred, or another process touched its archive path. Each dispatch now
records one current-state snapshot and publishes only the exact claimed file to
done or failed, so later work stays blocked until the claimed item has a
truthful outcome.

### Complete the combined daemon repair program

A roll-up line once tracked several related daemon repairs alongside their
individual tickets. Recovery, prompts, output, archive handling, watcher
warnings, fix-only, safe stopping, and landing-debt behavior each have an
accepted child record in this section. This is a summary, not a code defect.

### Warn when a sent message has no live watcher

A send could succeed into a mailbox no watcher was reading, leaving the user to
assume work had started. Send and ping now name that mailbox and any other live
watched mailbox without rerouting or failing the send.

### Protect fix-only mode and classify Sol tickets

Fix-only mode could not reliably distinguish a known repair from a new
discovery, especially near the discovery-demand limit. Public Sol requests now
declare closure or discovery, and fix-only launches only accepted closure work
plus the exact internal transport ping, so ambiguous actions fail before
launch.

### Provide regular windows for stopping the watcher

A busy watcher gave the user only occasional idle moments to stop it without
interrupting active work. After five completed role runs or fifteen
continuously busy minutes it stops starting work, waits for started work, and
prints a 19-to-0 Ctrl-C window.

### Stop the watcher after a chosen number of cycles

The user needed a planned stopping point instead of catching one short Ctrl-C
window. Positive `--cycle N` stops after N completed safe windows; zero waits
until no enabled message and no open backlog ticket remain. Missing or changing
backlog files keep zero mode running rather than closing early.

### Request landing when uncommitted work grows too large

A long-running branch could collect too much accepted work without asking the
Architect to land it. Above 400 changed lines the watcher creates one durable
landing request for that continuous episode and prints the debt in each demand
report, rearming once the branch returns to 400 lines or fewer. One repository
lock stops Fable and Sol landing concurrently.

<a id="closed-implementer-context-handoff"></a>
## Preserve exact Implementer context before a replacement session

LOW new functionality: repeating an unsuccessful approach wastes tokens, but
the old behavior corrupted no scientific output and erased no repository work.
An Implementer can reach its context limit mid-ticket; files survive, but a
fresh Implementer also needs to know what succeeded, what failed, and which
rejected approach would waste time if repeated. The automatic `PreCompact` hook
now asks for one small record: ticket, base and current commits, candidate
status, completed work, failures, rejected approaches, uncommitted files, and
next action, and the watcher verifies it against the current worktree instead
of inventing a summary.

<a id="closed-implementer-shared-notes-grant"></a>
## Let the Implementer open the shared notes directory it is told to use

HIGH bug fix: a dispatched Implementer could neither read its directive nor
write its return, so the role did no work and a live ticket turn produced
nothing. The failure was certain rather than likely: every Ollama Implementer
dispatch failed the same way.

The three roles work in separate worktrees, but the Architect worktree's
`ai/notes` is deliberately shared: it holds the mailbox, the current directive
note, and the guard programs, and every dispatched turn is pointed there
through `MAILBOX_SHARED_NOTES`. The launch command granted access to that
directory to the Red Team alone. A Claude Implementer never noticed, because
Claude Code opens an absolute path outside its working directory anyway; an
Ollama-served Implementer refused. Both Implementer launch commands now carry
the same `--add-dir` grant.

**Remaining:** a live Ollama Implementer dispatch has not been run again since
the repair; the next ordinary ticket exercises it.

## Repository organization and release hygiene

### Move study helpers into `emulator/studies/`

Four modules used a repeated `study_` filename prefix in the main emulator
folder. They now form one `emulator.studies` package, making their relationship
visible without changing scientific behavior.

### Keep operational backlog files out of release history

Temporary backlog and mailbox records could mix with the eleven durable notes
and enter a release commit. The repository now ignores operational records,
tracks exactly eleven permanent notes, and represents accepted work with one
reviewed commit. Landed as `c91791a`; the v1.0beta1 hygiene prerequisite is
satisfied and no release tag was created.

### Move all AI-development support under `ai/`

Tests, notes, gates, and tools were scattered among several root folders and
old entry points. `ai/README.md` is now the single starting point with the four
support folders only under `ai/`; no old root directory, wrapper, duplicate
entry point, or compatibility link remains.

**Remaining:** the real-data full-board run is in
[the workstation ticket](backlog.md#open-workstation-debt).

<a id="open-router-primary-schema-three"></a>
## Let the router read the current primary-worktree record

The daemon creates the saved Architect worktree with state schema 3 and
topology `separate-role-worktrees-v1`, but the router expected the retired
schema, so a valid fresh setup could fail before a role read the authoritative
backlog. The mismatch was repaired while the ticket stayed listed Critical and
Open, because landing did not require the Architect to close and seal the exact
ticket first.

Commit `864b69e` teaches the router the current record. Commit `9f99d1a` makes
a Closed, sealed ticket a prerequisite for a new landing: the daemon refuses GO
while the ticket is Open, preserves candidate C, and retires a rejected
same-cycle GO after a corrected GO lands. The closure gate is in
`ai/tools/mailbox_daemon.py`; the Architect instruction is in
`.claude/FABLE_ROLE.md` and `ai/notes/conventions-and-workflow.md`.

## Tickets closed while still filed under Open

These closed in place, before the archive existed, so their sections stayed in
the Open half of `backlog.md`. Moved here at the same 30 percent bar.

<a id="open-mps-test-import-isolation"></a>
## Isolate the matter-power adapter test without replacing imported modules

HIGH bug fix. The sigma-eight test replaced three `sys.modules` entries while
loading the Cobaya adapter; restoring the table left `emulator` submodules
attached to their parent package, so the suite became order dependent and could
report a false failure. The in-process loader is gone: the sigma-eight and
dark-energy adapter checks run in child processes that import through the
on-disk stand-in `ai/tests/cobaya_minimal_stub/`, placed first on the child's
PYTHONPATH. One negative control with a deliberately wrong known answer must
fail.

<a id="open-artifact-drift-import-isolation"></a>
## Test saved activation defaults without replacing a live function

HIGH bug fix. The drift gate changed `make_activation.__defaults__` in a
running process to prove that rebuilding reads the saved gate count, and a
shared function changed that way can leak into an unrelated test. The gate now
copies the emulator package into its temporary folder, changes only the
`n_gates` default line on disk, and rebuilds in a child process whose PYTHONPATH
names the copy first; the child refuses with a dedicated exit code unless the
changed default is live, so a launch that imported the ordinary package cannot
pass as proof. Helpers live in `ai/gates/checks/gsv_bitwise_drift.py`; the
durable behavior is `save-rebuild-drift.code-default-drift-ignored` in
`saved-emulators.md`. The full gate run on the workstation is owed
under [Complete older cross-family workstation
checks](backlog.md#open-workstation-debt).

<a id="open-finite-cycle-admission"></a>
## Make a finite watch start exactly the requested number of tickets

CRITICAL bug fix. With `--cycle 1` the watcher could start a second ticket while
the first waited for its Red Team review, then count the first cycle complete
and exit with that unrequested ticket already changed, defeating the human's
limit on runtime, edits, and model credits. Commit `20119a1` reserves finite
capacity before a public Architect turn, converts only an exact Implementer
ticket, and releases a valid non-ticket control outcome without counting a
cycle. A later request stays byte-for-byte untouched when the limit is full, and
the parent daemon owns candidate landing, restart recovery, push debt, and clean
role-baseline synchronization. Safe continuation after `main` legitimately
advances is [Recover safely when main advances after a landing is
prepared](#open-stale-landing-reaudit).

<a id="open-architect-note-landing"></a>
## Land Architect-owned permanent-note commits before later tickets use them

CRITICAL bug fix. Only the Architect may change the eleven permanent notes, but
no watcher operation moved a note-only commit onto `main`. Left on the
coordination branch, the next ticket either refuses to start from the newer
commit or starts from the old one, and the policy change never reaches the
candidate or GitHub. Commit `20119a1` adds the narrow Architect-only B/P landing
route with a restart journal, a bounded push-debt record, and a clean
role-baseline update. The route is cycle-free, unusable by the Implementer or
Red Team, and refuses to mix a permanent-note transition with an ordinary
ticket.

<a id="open-dataset-continuation-features"></a>
## Continue generated datasets exactly and manage old generations

MEDIUM new functionality. Asked for dataset continuation bitwise identical to
one uninterrupted longer run, plus a policy for pinning and retiring
generations. Retired with the publication framework it extended:
`compute_data_vectors/` returned to plain files under `chains/`, so there are no
generations to pin, and append draws from a stream derived from the seed plus
the existing row count, reproducible, and it never repeats a saved row. Exact
continuation would have required persisting complete sampler state.

<a id="open-getdist-column"></a>
## Write the GetDist posterior column with the correct meaning

MEDIUM bug fix. GetDist expects column two to hold the negative log posterior;
the generator wrote the ordinary log posterior under the name `lnp`, so
downstream analysis could reverse which of two samples had the better posterior.
Column two is now `minuslogpost`, the derived `chi2*` is `2 * minuslogpost`
(numerically identical to the old `-2 * lnp`), and a uniform draw writes a
neutral zero instead of a fabricated value.
`ai/tests/test_generator_posterior_column.py` loads a two-row chain with GetDist
and asserts the better-posterior row ranks better; a companion test proves the
old sign reverses that ranking.

<a id="open-power-zero-gradient"></a>
## Preserve the power activation gradient at zero

MEDIUM bug fix. `sign(x) * f(abs(x))` gave the right forward value but a zero
derivative at exactly zero, so zero-initialized layers and padded coordinates
could not begin learning while ordinary prediction checks looked correct. Both
production activation classes now compute the signed power as `x` times an even
magnitude ratio with analytic limit one at zero: the direct quotient away from
the origin and a quadratic series below `|x| = 1e-3`, with substituted inputs so
no unselected branch poisons a gradient. Constructors validate finite positive
`p_min < p_max`. With the origin derivative exactly one, the power families left
`ZERO_DERIVATIVE_HEAD_ACTIVATIONS` (now `relu` alone), so a power head pin is
accepted and a frozen trunk moves power CNN and transformer heads. Whole-model
CPU forward costs 1.15x the sign form. The GPU acceptance leg is owed with the
other workstation runs.

<a id="open-adapter-contracts"></a>
## Make every Cobaya bridge check inputs and protect cached results

HIGH bug fix. The five Cobaya adapters differed in their request checks and
several getters returned arrays backed directly by an internal cache, so a
request could be routed with the wrong segment, or one caller could mutate the
scientific result a later caller receives. Commits `d146590` and `5e0792a` give
all five one strict input and path contract, validate their family-specific
request and artifact facts, publish scalar results through Cobaya's derived
mapping, and return owned public arrays; the follow-up binds the gate to the
exact adapter source surface.

<a id="open-cmb-covariance-transaction"></a>
## Publish CMB covariance files without overwriting a good result

HIGH bug fix. A rerun or interruption could destroy an earlier valid covariance
matrix, or expose a half-written archive at the filename later calculations
read. Commit `4e4e09f` writes a hidden file and closes it before one
non-overwriting hard link gives the archive its final name; an existing name
keeps its contents, and an occupied output name stops the run before YAML
parsing or CAMB work. Hidden debris left by an uncatchable process kill is
parked below Low as [Remove hidden covariance files left by forced process
termination](backlog.md#parked-cmb-covariance-cleanup).

<a id="open-training-selection-record"></a>
## Record which saved weights the training run chose

MEDIUM bug fix. Training compares the untouched model, epoch snapshots, and
moving-average snapshots, but the loop returned only histories, so each driver
reconstructed the winner afterwards and could name a trained epoch when the
baseline won: one emulator file beside a report describing another candidate.
`training_loop_batched` now returns a validated selection record (candidate kind,
pass-local epoch, raw or EMA weights, and the winner's fractions, median, and
mean), `run_emulator` stores each pass's record in the resolved recipe and
publishes one run-level `resolved_train["selection"]`, and `validate_thresholds`
performs the one-time shape, finiteness, and strict-order check. Both train
drivers, the shared tune objective, and the saved h5 attributes read the record.
This implements the design `training-stack.md` "Selection record" already
specifies.

<a id="open-optimizer-scheduler-protocol"></a>
## Reject unsupported training options before a run starts

MEDIUM bug fix. CUDA forced a fused optimizer without proving the chosen
optimizer supported it, and Apple half-precision training could start without
the protection that keeps very small gradients from disappearing, so an ordinary
device or optimizer choice could fail after an expensive run began. Every named
capability is now checked before construction: fused is forced only when the
optimizer's constructor accepts it, and an explicit `fused` on a class without
one is refused by name; LBFGS is refused because the loop steps with no closure;
`OneCycleLR` and `CyclicLR` are refused because the loop advances the scheduler
once per epoch after warmup; reduced precision on MPS is refused because MPS
autocast runs in float16 with no gradient-scaling policy implemented.
Persisting a scheduler-cadence field was declined: the cadence is a code-owned
constant, and the per-batch refusal removes the one way a run could follow a
cadence the record does not imply.

<a id="open-memory-planner"></a>
## Measure memory without changing the model and reserve capacity before allocation

MEDIUM bug fix. **CLOSED — the described repair is not worth building.** An
audit of the sizing path found no demonstrated failure behind any of the three
requested changes. The batch-term probe runs one dummy forward on zeros with
scoped saved-tensor hooks, and no current model family carries
batch-normalization running statistics or active dropout, so that forward
changes no model state and draws no random numbers. The parameter budget already
multiplies weight bytes by five and the probe measures the real autograd-saved
activations; the omitted index and bound buffers are kilobytes against that
padding. A capacity-token reservation before worker allocation would only convert
a visible out-of-memory failure into a queue wait. The one real remainder, a
future family with stateful-forward modules, is parked as [Guard the sizing
probe if a stateful-forward family is
added](backlog.md#parked-memory-probe-stateful-forward).

<a id="open-mps-request-contract"></a>
## Validate matter-power requests before a run starts

MEDIUM bug fix. `must_provide` noticed only the optional sigma-eight quantity,
so a malformed particle pair, nonlinear setting, redshift, or wavenumber
survived Cobaya setup and failed later inside a getter. It now validates every
`Pk_grid` and `Pk_interpolator` requirement while setup can still stop: only the
`delta_tot` pair is accepted, the nonlinear choice must be boolean, requested
redshifts must lie inside the stored z grid (z is never extrapolated), and
`k_max` must be servable: inside the stored grid for the raw grid, beyond it
for the interpolator only when the power-law tails are enabled. Each refusal
names the observed request, the stored bound, and the corrective action.

<a id="open-implementer-blocked-outcome"></a>
## Let the Implementer stop honestly when a ticket cannot proceed

MEDIUM new functionality. **CLOSED — the honest stop exists as the checkpoint
family, and the enum on top of it is declined.** `.claude/OPUS_ROLE.md` requires
a relayable `IMPLEMENTER_HANDOFF` for every stop; a mid-unit stop is titled
`CHECKPOINT` and carries the changed files, completed checks, unfinished work,
and the decision requested from the Architect. The specialized stops have their
own validated shapes: `BUDGET BLOCKED`, the digest-bound capability checkpoint
for a rejected subagent launch, and `CONTEXT HANDOFF`. The daemon routes the
return to an Architect checkpoint audit instead of retrying, preserves saved
checkpoints across a restart, and never treats a checkpoint commit as candidate
C. What remained was a five-value blocker-reason vocabulary with no mechanical
consumer: the daemon must treat every reason identically, and the Architect
reads the required free-text evidence regardless of the label above it.

<a id="open-stale-landing-reaudit"></a>
## Recover safely when main advances after a landing is prepared

MEDIUM bug fix. **CLOSED — the supported recovery is the ordinary path, and a
shortcut would weaken it.** When `main` moves under a prepared landing the
watcher refuses, preserves the candidate and the prepared commit, and exits
nonzero; the ticket that cycle served is still open, so restarting the watch
runs a fresh cycle against the new `main`. The Implementer redoes the work on
the actual new parent and the Architect audits a real complete candidate under
the same uniform rule as every other landing, at a cost of one repeated
Implementer turn for a rare event. The requested alternative (stale marking,
provisional re-integration onto the new parent, a bounded re-audit protocol,
replacement-landing binding, and real-Git witnesses for each scenario) is a
second acceptance route through the daemon's highest-trust code, in which the
Architect reviews only the interaction between an old GO and the intervening
commits instead of a complete candidate.

<a id="open-relay-log-identity"></a>
## Give every role run its own relay-log filename

Bug fix. Relay log names under `ai/notes/relay/` used the role name and a
one-second timestamp, so two quick runs of the same role inside one second could
choose the same path and the later run replaced the earlier evidence.
`reserve_dispatch_log_path` in `ai/tools/mailbox_dispatch.py` now takes the name
by exclusive creation and appends a two-digit suffix until a fresh name is
accepted. `ai/tests/test_relay_log_reservation.py` hands the reservation one
frozen stamp, with no clock mocking, and requires both same-second logs to survive
with their own contents.

<a id="open-candidate-circumvention-review"></a>
## Check an accepted candidate for workarounds around rejected instructions

New functionality. A capable Implementer needs no malicious intent to preserve a
rejected design under another name, weaken a test so the result passes, or move
denied behavior into a wrapper. The audit section of `.claude/FABLE_ROLE.md` now
carries the consolidated **CIRCUMVENTION CHECK**: five questions answered
against the exact base-to-candidate diff before any GO: prohibitions preserved
even through generated files and wrappers, no rejected design recreated under
another name, no optional route restoring denied behavior, no checker change
that lets this same candidate pass, and no evidence bound to a different commit.
Hard refusals for executable bits, symlinks, and large additions were declined:
Git prints every mode and type change in the same raw diff the audit reads, and
a legitimate candidate can contain an executable script.

<a id="open-control-plane-protection"></a>
## Protect control files and keep candidates from weakening their own audit

Bug fix. **CLOSED — the enforced boundary exists and is machine-checked.**
`ai/notes/role-contract.yaml` holds one complete protected list: a candidate may
not touch `CLAUDE.md`, `.gitattributes`, `.gitignore`, `.gitmodules`, the
tracked backlog and its guard files, or any path under `.claude/`, `.codex/`,
`ai/tools/`, `ai/notes/mailbox/`, or `ai/notes/relay/`, and the contract
reader's safety floor refuses a contract that drops one of those entries. The
eleven permanent notes carry the SHA guard, with the Architect administration
turn as the one legitimate update path. For candidate changes to `ai/tests/` or
`ai/gates/`, the gate-integrity screen treats an unnamed change to the gate
surface as tampering and the circumvention check asks whether a checker was
weakened for its own candidate. A separate fingerprint store holding trusted
copies of test drivers and tolerance policies was declined: the audited base
commit in Git is that trusted copy.

<a id="open-character-budget-planning"></a>
## Plan a limited ticket across code, documentation, and protected notes

New functionality. **CLOSED — the reminder guards a failure that already
recovers cleanly.** The proposal was an advisory split of a positive `--max`
across Python, README or LaTeX material, and reserved permanent-note work. When
a clean candidate exceeds the limit, the Implementer returns the exact
over-limit blocked handoff with its measured count and the Architect revises the
plan, so discovering a bad allocation late costs one returned turn, not lost
work. Estimating the whole landing before writing the directive is ordinary
planning the directive template already forces, because the acceptance checklist
must require the exact candidate's guard result to be within the limit.

<a id="open-change-risk-classification"></a>
## Use change risk as well as character count when choosing checks

New functionality. **CLOSED — the directive machinery already forces this
decision per ticket.** The proposal was a risk label beside `--max`, since a
short numerical-normalization change can be more dangerous than a much longer
documentation update. Choosing proportionate validation is an existing Architect
duty with concrete enforcement points: every changed Python path carries a hot
or cold classification, every applicable style-contract row is copied into the
directive with named evidence, and the Architect selects the acceptance commands
per ticket. A parallel label taxonomy would restate that duty as vocabulary
needing stable meanings, boundary adjudication, and examples.

<a id="open-normalized-implementer-output"></a>
## Normalize untrusted Implementer output before the Architect reads it

New functionality. **CLOSED — the prompt boundary already exists where it
decides anything.** Every dispatched turn places the trusted material first: the
daemon's banner, the role preamble naming the authoritative role file, and the
byte-unchanged common preamble. The raw mailbox body enters only after the exact
`--- MESSAGE ---` delimiter, as the prompt's suffix. Returns that carry
authority are structurally validated: the handoff contract checks envelope,
exact rows, and admission values, and the daemon acts only on parsed exact
headers, so instruction-shaped prose inside a body cannot become an action
whoever wrote it. The remaining proposal (a versioned report format with
Unicode normalization, terminal-escape scrubbing, and duplicate-field rejection)
is input sanitation proportioned to hostile bytes, not to a cooperating
Implementer.

<a id="open-authenticated-control-messages"></a>
## Authenticate trusted ticket and landing messages proportionally

New functionality. **CLOSED — the described defense answers a threat this
workflow does not face.** The protection target is an Implementer that bends
rules to make its own ticket pass, not an operator counterfeiting credentials.
Against that target the structural controls already close every forgery route
the ticket names: the daemon acts only on parsed exact envelope headers, never
on free text; an Architect GO is consumed only from the Architect lane's own
outbound and must bind the exact candidate commit; the daemon alone creates the
landing commit, after the deciding Architect process has exited; and the sealed
backlog plus full commit hashes bind every durable record to exact bytes. A
local secret-key layer would add no boundary, since the key would live on the same
machine where every role runs, while key creation, storage, rotation, and
recovery would become permanent control-plane complexity.

<a id="open-control-plane-regression-runner"></a>
## Run every required control-plane regression with one command

New functionality. `unittest discover` runs only the `test_*.py` modules, so
"the AI tests passed" could quietly omit the `*_repro.py` programs that rebuild
interruptions, stale landings, worktree failures, push debt, and restart
recovery. `python3 ai/tests/run_control_plane_regressions.py` is now the
acceptance command for changes to the AI workflow controller: it runs the
control-plane test modules and every reproduction from one explicit manifest,
each command in its own child process from the repository root, prints one
verdict line per command, writes complete output to a named log file, and
returns zero only for a complete pass. It exits 2 before any check when a
manifest entry is missing or duplicated, lacks its README inventory row, or when
a `*_repro.py` on disk is unlisted. There is no skip mechanism: a check that
cannot run is a failure.

<a id="open-daemon-authority-modules"></a>
## Reduce daemon risk through small authority-boundary extractions

New functionality. **CLOSED — the requested extraction exists in the tree.**
`mailbox_daemon.py` had grown to roughly fourteen thousand lines in which
provider calls, mailbox movement, candidate records, backlog closure, landing,
worktree synchronization, push recovery, and restart behavior all interacted.
The daemon is now a coordinator of about two thousand lines beside part files
split along the boundaries this ticket proposed: `mailbox_providers.py`,
`mailbox_dispatch.py`, `mailbox_store.py`, `mailbox_envelopes.py`,
`mailbox_cycles.py`, `mailbox_tickets.py`, `mailbox_landing.py`,
`mailbox_recovery.py`, `mailbox_worktrees.py`, `mailbox_watch.py`, and
`mailbox_control_plane.py`. Every cross-file reference routes through the
coordinator's namespace, so each repeated decision keeps exactly one owner. A
future extraction needs its own ticket with its own boundary.

<a id="open-github-push-choice"></a>
## Let the user choose whether accepted work is pushed to GitHub

New functionality. The watcher accepts `--github yes|no` with the documented
default `yes`, so existing commands keep pushing unchanged. The choice is read
in exactly one place, inside the push function that every landing kind and
every debt retry already calls, so the local landing path is byte-identical for
both values. With `no` the function returns before any Git command: nothing
contacts the remote, one sentence names the verified local landing and the user
choice, no push-debt record is written, and debt records from earlier runs stay
on disk. The watch banner states the choice at startup, and the daemon's
self-restart re-executes the original command line so the value survives.
`ai/tests/test_role_workflow_behavior.py` proves the skip contacts no remote
against a deliberately missing repository path, preserves an earlier debt file
byte for byte, keeps the default at `yes`, and rejects an unknown value at
command-line parsing.

<a id="open-landing-backlog-identity"></a>
## Bind each landing to its candidate and sealed backlog

New functionality. **CLOSED — Git already stores the answer the ticket wants
recorded.** "Which candidate and which exact backlog bytes produced this landing
commit" is answerable from the repository alone: durable landing state names
candidate C and landing L, and the exact backlog bytes are L's own tree, so
their digest is recomputable at any time with
`git show L:ai/notes/backlog.md`. The daemon verifies the sealed overlay before
it builds L, which is the moment verification can still refuse. Writing the
digest again into contract fields, commit trailers, and a saved tuple would
create a second record that can only agree with the tree or falsely disagree
with it.

<a id="open-backlog-sync-crash-cuts"></a>
## Test every interrupted backlog synchronization step

New functionality. **CLOSED — the test cannot be built without breaking a
stronger rule.** Stopping the process exactly between the synchronization
boundaries (after the move to recovery, after the tracked restore, after the
fast-forward, before the recovery-file delete) requires either replacing
`os.replace`, the trusted `git restore`, or `os.unlink` while the routine runs,
a monkey patch the Python contract prohibits in tests, or planting injection
hooks inside the production synchronization code, which changes the trusted path
to test it. Timing a kill from outside cannot hit those boundaries
reliably. The end states the cuts would produce are already covered: equal bytes
converge, conflicting bytes fail closed and stay preserved, and the guard binds
the exact accepted bytes.

<a id="open-ai-ticket-latex-guide"></a>
## Write a LaTeX guide to the AI ticket system

New functionality, held last until the user advanced it. The manuscript exists
as `documentation/cocoa_flow_guide.tex` with its tracked
`documentation/cocoa_flow_guide.pdf` (23 two-column pages), named after the
system's own name, CoCoA-Flow. The user directed it at teaching a reader to read
the `ai/tools/` code with the same discipline the emulator manuscript teaches,
and the delivered structure follows: a notation section for the commit labels C,
L, M0/M1, B/P, and D0/D1; an end-to-end ordinary-ticket walkthrough; one section
per module family; verbatim code excerpts with exact refusal messages; and a
staged file-by-file study route with three shorter role-specific routes. The
beginner-operator tutorial sketched in the ticket stays owned by `ai/README.md`
and `ai/tools/README.md`, which the manuscript names as its reading stage 0.
Build from the repository top with `latexmk -pdf
-output-directory=documentation documentation/cocoa_flow_guide.tex`; the
frontispiece path is repository-root-relative. `documentation/README.md` carries
the catalog row.
