# Red-team audit of the landed batch and didactic repair queue (2026-07-13)

This is the Red Team's independent review of the Implementer's handoff at
`bd60a9f`, first rechecked from the merged tree at `a93f417` and followed up
against `8e8e59b` below.  It is a
repair specification, not a claim that the listed work is already complete.
The Fable Architect's role and queue ownership are unchanged.

The reader standard for every documentation item below is a second-year
physics undergraduate encountering machine learning, PyTorch, HDF5, and this
emulator library for the first time.  A passage that is accurate but requires
the reader to infer an unstated Python mechanic does not pass.

## Evidence actually rerun

The following checks ran in the local CPU environment:

```
python3 -m compileall -q emulator gates compute_data_vectors *.py
python3 gates/run_board.py --list
PYTHONPATH=. python3 gates/checks/board_selftest.py
python3 gates/checks/generator_seed.py
```

`compileall` and `--list` passed.  `board-selftest` passed 33/33, and
`generator-seed` passed.  The local environment does not contain `torch`,
`psutil`, or `yaml`, so the new torch-importing checks were reviewed
statically rather than counted as executed.  Their live acceptance remains a
workstation obligation; absence of a local dependency is not a green result.

## Follow-up audit of the Implementer's four completed units at `8e8e59b`

This section supersedes the status, but not the historical evidence, in the
earlier findings below.  The Red Team read the diffs and current call paths,
then reran:

```
python3 gates/run_board.py --list
PYTHONPATH=. python3 gates/checks/board_selftest.py
python3 -m py_compile emulator/training.py emulator/experiment.py \
  gates/run_board.py gates/checks/board_selftest.py \
  gates/checks/cmb_identity.py gates/checks/finite_contract.py
```

`--list` returned zero and showed the 40 registered gates.
`board-selftest` passed all 39 reported assertions.  The raw-log tests drive
the real `_resume_state`, including valid evidence, truncation, deletion,
missing digest, byte edit, dependency refusal, and a load-bearing tamper arm.
The local Python still has no torch, so the optimizer and CMB schema gates
remain workstation evidence rather than local greens.

### RT-IMPL-01 — raw-log resume trust is accepted on the CPU path

Commit `b6cfd87` uses one `_log_stale` predicate from `_resume_state`.
Execution skip, dependency acceptance, `--list`, and `BOARD.md` all obtain
their verdict through that state.  A stale log cannot remain a current PASS.
No repair is requested for this unit.

### RT-IMPL-02 — executable-surface preflight is reopened on one exact exclusion

Commit `7f69c35` correctly adds `compute_data_vectors/`, `cobaya_theory/`, and
`syren/` to the dirty-tree watch.  It does not implement its stated
`board_config.json` exclusion.  `_EXECUTABLE_DIRS` contains the directory
`"gates"`, and the executed command is equivalent to

```
git status --porcelain -- emulator gates compute_data_vectors \
  cobaya_theory syren <root drivers>
```

`gates/board_config.json` is a tracked file below `gates/`, so Git includes it
in that pathspec.  The comments at `gates/run_board.py:716-727`, the config
dump text near line 1242, and `notes/gates-and-board.md` all say it is
excluded.  A portable local deployment override therefore makes preflight
red even though the documented contract says it may vary.

Required repair:

- construct the watched pathspec with an explicit exclusion for exactly
  `gates/board_config.json`; do not exclude the rest of `gates/`;
- add a preflight self-test that modifies only that file in an isolated fake
  or temporary Git tree and proves the dirty-tree decision stays green;
- in the same test, modify `gates/board.py` and prove the decision turns red,
  so an over-broad exclusion cannot pass;
- keep the existing generator, adapter, syren, and root-driver probes; and
- make the printed watched surface derive from the same list the command uses.

This is an amendment to the completed preflight unit, not a new parallel
manifest design.  The per-gate executable/input manifest remains separately
open.

### RT-IMPL-03 — typed numeric schemas are structurally accepted; live lane owed

Commit `c9ace04` wires `_validate_optimizer_opts` into both optimizer
factories and validates CMB `as_ref`/`tau_ref` before range comparison.
Boolean and numeric-string coercion is removed at those paths.  The pure
control and red-leg sets cover the values named in the repair spec.  The
workstation still owes the board-listed finite-contract Part J path and the
post-step parameter/optimizer-state check; source review cannot promote those
lanes to green.

The commit introduced the same semantic helper twice, at
`emulator/training.py:168` and `emulator/experiment.py:701`.  This duplication
does not invalidate the schema repair, but it belongs in the code-ownership
cleanup below so later finite-real rules cannot drift.

### RT-IMPL-04 — README factual repair is accepted

Commit `dce3d69` removes the hard-coded 32-gate count, the selector
warn-and-proceed claim, the public internal ledger token, and the retired
grid2d full-materialization blocker.  An untruncated scan of both READMEs found
no residual instance of those exact stale claims.  The larger first-time-user
documentation campaign remains open and must not be conflated with this
bounded factual correction.

## Binding rulings for the structured evidence-map rollout

These decisions answer the five questions in
`red-team-implementer-handoff-2026-07-13.md`.

1. Keep both fields.  `Gate.maps` is the short human promise.  `Gate.evidence`
   is the machine-checkable list of individual acceptance legs.  The rollout
   must make them agree; neither replaces the other.
2. Validate the evidence registry on every invocation, including `--list`.
   Listing an invalid registry as if it were runnable would be a false
   advertisement.
3. Keep explicit `<a id="..."></a>` markers.  A deliberately named marker is
   stable when a heading is reworded.
4. Do not use an internal audit-code prefix in an assertion identifier.
   Name a leg `<gate-id>.<plain-leg-name>`, for example
   `board-selftest.exit-truth`.  The raw log must teach a reader what ran;
   internal project codes remain in `notes/`.
5. Require the note named by each assertion anchor to equal the gate's
   `home`.  A gate may link to other notes in human prose, but its executable
   acceptance legs belong to one declared home.

One additional rule is required for the rollout to prove execution rather
than registration.  Every declared assertion id must emit exactly one
terminal result for a run: `PASS`, `FAIL`, or an explicit non-green
`UNAVAILABLE`.  The runner compares the declared and executed id sets.  A
missing id, a duplicate id, an unknown emitted id, a check-script crash before
its manifest, or a conditionally omitted leg makes the gate red.  External
check scripts therefore need a small machine-readable assertion record; the
gate's aggregate subprocess exit code is not enough.

The Fable Architect ratified these rulings without modification and added one
verdict constraint: when the module's own acceptance table defines one
in-code verdict, the rollout records one verdict rather than manufacturing a
second parallel verdict from the same execution.

## Correctness review of the landed batch

### Critical reopen: resume can trust altered or incomplete evidence

`gates/run_board.py::_resume_state` checks the stored code and input digests,
but it never checks `log_digest`.  `_log_digest_mismatch` only annotates
`BOARD.md`.  Deleting, truncating, or editing a raw log can therefore leave
the corresponding status as a current `PASS`, and a normal rerun skips it.

Required repair:

- Make a missing digest, missing log, or mismatched log digest a non-green
  resume state such as `stale-log`, and rerun a selected gate.
- Add self-test arms for truncation, deletion, missing stored digest, and the
  valid unchanged control.  Each arm must call the same resume decision the
  real runner uses.
- The displayed board and the skip decision must consume the same state; a
  warning-only path is not sufficient.

The two other resume digests are not the surfaces their documentation claims:

- `_gate_code_digest` hashes `inspect.getsource(gate.run)` and literal
  `gates/checks/*.py` files mentioned in that function.  It omits shared
  helpers in `gates/board.py`, runner behavior in `run_board.py`, imported
  production modules, and transitive imports.  Changing a shared helper or the
  code under test can therefore leave a stored `PASS` current.
- `_gate_input_digest` hashes every YAML in `yaml_dir`, so an unrelated YAML
  edit stales every gate.  It does not establish a manifest of the particular
  data, artifact, covariance, and axis files the gate consumes.

The replacement is a proposed, reviewed executable/input manifest.  Hash the
runner and shared gate helpers, each invoked check script, and the production
modules that the gate declares.  Hash only that gate's resolved YAML and
external file inputs.  Persist the manifest beside the digest so a reviewer
can see why the gate became stale.  A digest with no inspectable membership is
not evidence.

Preflight has the same surface hole.  Its dirty-tree watch covers
`emulator/`, `gates/`, and root Python drivers, but not
`compute_data_vectors/`, `cobaya_theory/`, or `syren/`.  The executable
surface must be defined once and shared by dirty-tree preflight and code
digesting.

### Critical reopen: host-RAM availability changes the seeded training order

The byte estimate now counts parameter, target, and reindex arrays.  Two
acceptance details remain open:

- The banner prints `params + dv = total`, but `total` also contains
  `idx_bytes`.  The displayed arithmetic is false.  Print all three named
  terms, their exact byte total, budget, comparison operator, and selected
  branch.
- The promised exact-fit boundary is absent.  The code uses `need < budget`;
  the gate needs below, equal, and above cases so equality is a deliberate
  policy rather than an accidental branch.

The row fixture hides a reachable representation difference.  The normal
selected index is a unique, generally unsorted prefix of a seeded permutation.
The resident branch sorts it before copying and returns local `arange`; the
disk-backed branch returns the original shuffled `idx`.  The training loop
later executes

```
perm = tidx[torch.randperm(ntrain).numpy()]
```

on whichever sequence it received.  The same epoch permutation therefore
maps to different cosmologies when the RAM decision changes.  For selected
global rows `[9, 2, 5]` and an epoch permutation `[1, 0, 2]`, the resident
representation executes global rows `[5, 2, 9]`, whereas the disk-backed
representation executes `[2, 9, 5]`.  With batch size two, even the first
minibatch contains a different pair.  Host-memory availability can therefore
change training under the same seed.

The current gate sorts and deduplicates the disk indices before comparing, so
it proves set equality and hides the defect.  The module's duplicate example
is also factually wrong: for `idx=[9,2,9,5]`, the disk branch returns that
original sequence, not `[2,5,9]`.

Required repair: preserve one canonical selected-row order across both
representations.  If compact storage is kept in sorted disk order, return an
explicit local-coordinate array that maps the original selected order into
that compact storage rather than replacing it with plain `arange`.  Then drive
the real loader and epoch permutation in both storage regimes.  Prove that
parameters, targets, sibling-dump rows, minibatch membership, and minibatch
order are identical under the same seeds.  Include a mutation arm that
restores `arange` and must fail.  Do not make both arms look equal by sorting
inside the assertion.

### Reopen: optimizer schema still coerces invalid public values

`_validate_optimizer_opts` converts controls with `float(...)`.  Consequently
booleans and numeric strings can be admitted as learning rate, weight decay,
epsilon, or betas.  This conflicts with the repository's public-boundary rule:
configuration values are validated by type, not made valid by coercion.

Require finite, non-boolean real numbers before any conversion.  Keep the
documented domains: positive learning rate and epsilon, nonnegative weight
decay, and betas in their allowed interval.  Add red legs for `True`, `False`,
numeric strings, NaN, infinity, and each endpoint.  The separate post-step
finite check for model parameters and optimizer state remains workstation
owed; the schema half does not close it.

### Partial: artifact, family, command-line, generator, and diagnostic gates

- `artifact-readback` proves `_read_native_bool` in isolation.  Its scalar
  `rescale` check merely searches one source file for the strings
  `"rescale":` and `"none"`; it does not prove they form the saved field.
  The real save, forge, rebuild, and fine-tune parity legs remain required.
- `family-first` is sound at the full public configuration boundary, which
  later rejects mutually exclusive family blocks.  The direct helper test
  does not try a mapping containing both its expected block and another
  family block.  Add the public-boundary leg so the advertised exactly-one
  claim is tested where it is enforced.
- Strict command-line parsing is present.  The live representative imports
  remain workstation/dependency owed.  `emulator/cocoa.py` still contains
  stale prose referring to work done “before parse_known_args”; remove the
  retired mechanism from documentation.
- The owned generator RNG is present, and the local pure check passed.  The
  append replay and worker-count invariance checks remain open, as the
  Implementer recorded.  The source-text census is not a substitute for a
  generated-file equality test.
- The shared diagnostic score screen is structurally present.  The final
  producer census in `diagnostics_domain.py` should be an AST call-site check,
  not a broad substring condition that can pass for the wrong reason.
- The CMB amplitude formula and persisted reference facts are structurally
  consistent.  `validate_cmb` still applies `float(...)` to `as_ref` and
  `tau_ref`, so booleans and numeric strings remain admissible.  Use the same
  finite non-boolean real predicate as other public scientific controls.

## README audit: current text that is already false

The README layer must be corrected in the same implementation batch as the
code or gate it describes.  The current tree contains these direct
contradictions:

- `gates/README.md` and `emulator/README.md` still call the board a 32-test
  board.  The registry has 40 gates.
- `gates/README.md` says unknown selectors warn and proceed.  The landed
  runner now rejects them as a nonzero usage error.
- The gate README says one helper captures every command, while 27 external
  check scripts produce their own internal reports and the runner sees only
  their aggregate exit.  Describe the current two-layer mechanism and the
  open declared-versus-executed reconciliation honestly.
- `emulator/README.md` still says grid2d staging materializes the unthinned
  float64 selection as an open blocker.  Bounded grid2d staging has landed;
  its remaining review issue is lifecycle/evidence, not the retired full
  materialization claim.
- `emulator/README.md` calls the corrected Simpson path “45M-12” in public
  prose and repeatedly uses dated “ruling” language.  State the executed
  numerical rule and point to the topic note without internal codes.
- Root and package READMEs mix current behavior, open defects, and run-history
  anecdotes in the same table cells.  Use a stable role sentence in the file
  map and a separate, clearly dated “current limitations” table.  A first-time
  reader should not have to decide which clause is architecture and which is a
  temporary audit status.

The root README is substantially more didactic than the in-file prose: it
defines whitening, dense layers, residual blocks, PCE, and activation
families.  Preserve that section-by-section teaching shape.  Do not replace
it with a compressed file inventory.

## In-file documentation audit for a first-time ML reader

The Fable Architect's broader untruncated scan of `emulator/` and `gates/`
found 108 lines in about 25 Python files containing internal unit numbers,
dated rulings, reviewer biography, or design-ledger identifiers.  The pattern
family includes `45M`, `unit N`, `increment`, `Architect`, `Implementer`,
`ruling`, `adjudicat-`, `D-*`, and `POL-*`.  Examples include `unit 60`,
`unit 14(f)`, “Architect confirms,” and dated “symmetry ruling” comments.  The
prior cleanup searched only for the literal `45M` token and therefore did not
satisfy the no-internal-bookkeeping rule.

Required cleanup:

- Replace every internal code, queue number, reviewer name, and adjudication
  anecdote in Python docstrings, comments, errors, report labels, and board
  descriptions with the current mechanism and reason.  Identifiers required
  by executable compatibility may remain as identifiers, but human prose
  around them must use plain language.
- Completion is an untruncated zero-hit scan over the full pattern family,
  not only `45M`.  Record the exact patterns, scope, and a reviewed allowlist
  of identifiers whose spelling is required by executable compatibility in
  the gate output.
- Correct existing punctuation defects while touching those passages, such
  as `unit 14(f),, clause 4` and the duplicated score-boundary sentence in
  `gates/board.py`.

The problem is deeper than missing docstrings.  The AST census reports 57
public methods/classes without docstrings in `emulator/`, 125 in `gates/`,
and 14 long gate checks whose docstring is fewer than eight lines.  Tiny test
doubles and one-line `forward` methods do not each need a ceremonial block.
Prioritize these teaching owners:

1. `emulator/data_staging.py`: rewrite the row-coordinate story from one
   concrete unique selection.  Define global disk row, compact resident row,
   loader row, eager advanced-index copy, memmap view, and the purpose of
   `dump_rows` before showing code.  The current comment is longer than the
   operation yet still makes a false claim about the disk branch.
2. `emulator/experiment.py`: correct the lifecycle.  Staging builds host
   source dictionaries, not on-device loaders.  `build_geometry` constructs
   geometries and the scientific loss, not the model.  Saving writes `.emul`
   weights and an `.h5` recipe/geometry record, not the reverse.  Add “state
   before / action / state after” blocks to `from_config`, `stage_train`,
   `stage_val`, `build_geometry`, `train`, and `run`.
3. `emulator/warmstart.py`: keep the tensor diagram, but add one named-column
   example and define every slice as view or copy.  Define `torch.no_grad`,
   packed targets, zero-padding direction, dtype/device placement, and why
   parity is checked before training.
4. `emulator/diagnostics.py`: only `coverage_diagnostic` returns a boolean
   verdict.  The module currently says two functions do.  A local-linear fit
   is one comparator, not a mathematical lower bound, “best smooth method,”
   or proof of data-only hardness.  Replace those claims and name the status
   returned when a statistic is unavailable.  Define and justify neighbor
   counts, thresholds, deciles, and numerical floors at their use sites.
5. `emulator/results.py`: define an HDF5 file, group, dataset, and attribute;
   define `detach`, `cpu`, `numpy`, recursive group writing/reading, dynamic
   class import, and strict state loading.  Provide separate ownership trees
   for ordinary, PCE, transfer, and refined artifacts.  State clearly that
   the two files are currently written sequentially and are not yet a
   digest-bound transaction.
6. `gates/board.py`, `gates/run_board.py`, and `gates/checks/*.py`: every
   scientific check needs a short sequence: claim, real production boundary,
   fixture, independent expected value, valid control, mutation arm, and
   printed observation.  Define stub, fake, monkeypatch, manifest, digest,
   skip, stale, and dependency before using them.  Avoid audit history as an
   explanation of mathematics.
7. `emulator/activations.py` and `emulator/designs/*.py`: the public
   constructors and `forward` paths should name input/output shapes,
   broadcasting axes, registered parameter ownership, whether an operation is
   in-place, and why zero-initialized heads require a nonzero derivative at
   zero.  These are central learning files, not self-explanatory PyTorch.
8. `emulator/geometries/grid.py` and `grid2d.py`: replace an unglossed “law
   space” with the actual stored formula at each public boundary.  State the
   flattening rule, axis order, dtype conversion, and which transform is
   reversed during prediction.

The detailed reread found four documentation defects that a docstring census
alone cannot detect:

- `emulator/activations.py` says that the bounded power families “never blow
  up,” “keep any real p finite,” and keep the output finite.  Constraining the
  exponent prevents an optimizer from choosing an arbitrarily large power;
  it does not make finite-precision overflow impossible.  A sufficiently
  large finite `float32` input can still overflow a mildly super-linear
  power.  Define the actual safety property (positive power base and bounded
  exponent), state the accepted finite input domain, and leave nonfinite
  detection to the executable finite contract.  Also replace “your H” and
  “the paper's H” with the formula or the class name at the point of use.
- `emulator/data_staging.py::stream_stats` teaches that the one-pass
  `sum(x)`/`sum(x**2)` form “keeps precision and avoids overflow.”  The queued
  stable-moments repair exists precisely because cancellation can destroy the
  variance and squaring can overflow.  Until that repair lands, the current
  algorithm and limitation must be named honestly; after it lands, teach the
  Chan/Welford merge rather than retaining the retired explanation.
- `emulator/designs/blocks.py` explains that grouped variants “were tried and
  removed (see git history).”  That history does not tell a first-time reader
  what the current tensor layout is.  Replace it with the present ownership
  rule: which axis is a token, which axis is a feature, and which module owns
  the correction.  Comments such as “your H” likewise become the executed
  formula and shape.
- `emulator/geometries/grid2d.py` uses emphatic all-caps and audit chronology
  (“ANY law,” “WHOLLY,” “dead dump,” “stale generator,” and “pre-pin
  byte-identical”) where definitions are needed.  Define the stored surface,
  the row-major flattening order, the low-k prefix mask, the base-law
  composition, and the exact inverse operation.  State current limitations in
  a labeled paragraph rather than embedding reviewer history in mechanics.

The gate checks need the same treatment.  For example,
`gates/checks/finite_contract.py` contains a long module chronology but gives
several 80--150 line `check_*` functions one sentence, while
`gates/checks/cmb_identity.py` mixes physical claims, fixtures, retired rules,
and audit dates in one large bullet list.  Each check function must teach the
local experiment in the same order: input tensor shapes; real production
function called; independently calculated expected value; valid control;
deliberately broken mutation; and the value printed on failure.  A reader
should not have to search a project ledger to learn what “Part H,” “the
ruling,” or a fake object is intended to prove.

Every rewrite must preserve code behavior.  For documentation-only commits,
strip docstrings and comments from the AST before and after, compile the
files, and scan for stale phrases.  Behavioral changes discovered during the
rewrite become separate implementation units with their own gates.

## Code-ownership and bloat review

This review does not use line count as a target.  Shortening comments,
putting several function arguments on one line, replacing named loops with
dense NumPy indexing, or inlining a one-use teaching helper would fail the
user's readability rules.  A consolidation is justified only when several
places own the same behavior or when a function is unreachable.

The findings in this section were produced after the Fable Architect returned.
They are submitted to the Architect for adjudication.  The Implementer does
not start them from this note alone, and the Architect retains queue-number
ownership.

The AST census at `8e8e59b` measured:

| Tree | Python lines | Functions/methods | Classes |
|---|---:|---:|---:|
| `emulator/` | 25,478 | 447 | 48 |
| `compute_data_vectors/` | 3,268 | 62 | 5 |
| `cobaya_theory/` | 1,605 | 55 | 6 |

A repository-wide token-reference pass found no module-level function in
these trees that appears only at its definition.  Dynamic class loading means
that even this scan could not prove deletion by itself.  There is therefore
no accepted bulk-deletion list.

### BLOAT-01 — `emulator/`: one small consolidation, no broad compression

The new `_is_finite_real` implementation is duplicated in
`training.py:168-186` and `experiment.py:701-720`.  Give the predicate one
owner and import it at the other boundary.  Preserve its full didactic
docstring at the owner and use a short comment at the consumer naming why a
boolean is not a numeric control.  The unit's existing schema legs must run
unchanged afterward.

Two unused-import candidates are mechanically supported:
`validate_loss` in the import list at `experiment.py:152` and `torch` at
`losses/scalar.py:23`.  Remove them only in a bounded cleanup with import,
compile, and gate checks; they are low priority and do not justify a separate
campaign.

Keep the following even where a lexical scan shows only one call:

- plotting helpers such as `_coverage_panels`, `_floor_panel`,
  `_hard_direction_panels`, and `_save_pages`; their names divide a long
  figure workflow into teachable operations;
- per-class `encode`, `decode`, `from_state`, and `chi2` methods; these are the
  explicit geometry/loss protocols and often carry different physical
  owners despite similar bodies;
- family-specific `attach_head_coords` methods; their persisted axes are not
  interchangeable; and
- factory functions such as `make_scalar_chi2`; the call site should continue
  to say which constructible object is being made.

The very long `build_geometry`, `from_config`, `run_emulator`, and
`training_loop_batched` functions are review hotspots, but their length does
not prove redundant behavior.  If they are decomposed, extract named state
transitions with one argument per line.  Do not reduce total lines by hiding
family branches in anonymous dictionaries, clever array expressions, or
metaprogramming.

### BLOAT-02 — `compute_data_vectors/`: one multi-array store owns 449 repeated lines

The background, CMB, and MPS generator subclasses each reimplement the same
seven storage hooks:

```
_dv_chk_files  _dv_load_chk  _dv_save  _dv_append
_dv_alloc      _dv_write     _dv_zero
```

Those suites occupy 152, 143, and 154 lines respectively.  Their repeated
mechanics are RAM accounting, RAM-versus-memmap selection, temporary-file
publication, row append, row write, row zeroing, and row/width validation.
Only the ordered quantity names and widths differ.  This is genuine duplicated
ownership: a payload-finiteness or transactional fix can land in one family
and be missed in the other two.

Required design proposal before editing:

- keep the current single-array defaults in `GeneratorCore` for the lensing
  driver;
- introduce one plainly named multi-array store owner in
  `generator_core.py`;
- let a family supply an ordered list of quantity names and an explicit width
  for each quantity;
- use ordinary `for quantity in quantities` loops for allocation, validation,
  append, save, write, and zero operations;
- keep family physics, sidecar creation, and `_compute_dvs_from_sample` in the
  family driver;
- do not encode CMB's array payload and background/MPS dictionary payload in
  a dense indexing trick--use one explicit payload accessor whose arguments
  are named; and
- preserve one argument per line for long signatures and calls.

Acceptance must exercise the real public generator path for all four
families: new allocation, checkpoint reload, append, forced RAM path, forced
memmap path, failed-row zeroing, nonfinite/wrong-shape refusal before success
is recorded, exact sidecar retention, and byte-identical valid payloads.
Append replay and worker-count invariance remain workstation evidence.  This
refactor should follow the active generator-integrity repairs so it consumes
their final contract instead of racing them.

The driver files also contain small safe cleanup candidates: unused `sys` and
`traceback` imports in the background/CMB/lensing drivers, unused `traceback`
in MPS, and unused `sys` in `compute_cmb_covariance.py`.  Batch those with the
store consolidation or documentation cleanup; do not create a unit for six
imports.

### BLOAT-03 — `cobaya_theory/`: centralize mechanics, retain family lifecycle prose

Every adapter contains an exactly identical 19-line `_pick_device` body.
Every adapter also repeats the unknown-`extra_args` loop, ROOTDIR-relative
path expansion, and `bool(extra_args.get("compile"))`.  The repeated block is
already a correctness boundary: the queued adapter-contract work requires
typed booleans, explicit roots, and non-coercing requested values.

Consolidate these mechanics when that boundary lands:

- one helper validates `device` as an allowed string and resolves availability;
- one helper validates `compile` as an actual boolean;
- one helper rejects unknown option keys while accepting the family name and
  its plain-language retired-key explanation for the error;
- one helper resolves an absolute artifact root and refuses a missing ROOTDIR
  for relative roots; and
- all five adapters use those helpers before loading an artifact.

Do not generalize the five `initialize` or `calculate` methods into a large
parameterized base-class template.  Their spectrum assembly, scalar-name
publication, redshift windows, and matter-power composition are different
scientific operations.  Likewise, keep the one-line `get_requirements`
methods visible: they are Cobaya protocol methods, not needless wrappers.

The consolidation gate calls the public lifecycle of all five adapters and
includes a mutation arm that restores one local `bool(...)` or `str(...)`
coercion.  It also scans for zero remaining local `_pick_device` definitions,
so a sixth rule cannot appear by drift.

### BLOAT-04 — documentation in the two newly reviewed trees

The four `dataset_generator_*.py` files place their long introductions after
imports as comments, so Python exposes no module docstring for them.  Move the
introduction before imports and rewrite it as a present-state module
docstring.  Define rank zero, worker rank, MPI message, checkpoint, memmap,
append, and sidecar before the implementation uses those words.  Replace
audit-history prose such as “MOVED VERBATIM,” “DONT REMOVE THIS,” and
“SOME WEIRD BEHAVIOR” with the current mechanism and a reproducible condition.

The Cobaya adapter module docstrings are stronger, but their repeated helper
comments still assume prior framework knowledge.  Each adapter's class
docstring should define the four lifecycle times used in the manuscript:
construction, requirement negotiation, one-sample calculation, and getter
readback.  Define `state` as Cobaya's mutable result dictionary for one
sample, not as model weights.  Keep the vendored interpolator attribution,
but state the local validation and extrapolation differences next to the
methods that implement them.

Reformat long public signatures so each argument occupies its own line.  This
is a readability change, not a line-reduction target.  Documentation-only
commits carry AST-with-docstrings/comments-stripped identity, compile checks,
and untruncated scans for the retired phrase families.

## Ordered Implementer queue from this review

1. Submit preflight finding `RT-IMPL-02` to the Fable Architect: the executed
   pathspec does not implement its stated exclusion.  If adjudicated in,
   exclude exactly
   `gates/board_config.json` from the dirty-tree pathspec and prove both the
   exclusion and a nearby watched control.  Then propose the reviewed
   per-gate executable/input manifest.  Raw-log resume trust is closed on the
   CPU path; do not reopen it without a counterexample.
2. Staging truth: make the seeded training/minibatch order independent of the
   RAM branch, correct the banner, define equality policy, and prove the real
   downstream sequence in both storage regimes.
3. Evidence rollout: apply the rulings above, then reconcile declared and
   executed assertion ids across all gates and check scripts.  This follows
   staging truth under the Architect's re-pinned order.
4. Complete the post-step finite workstation gate.  The optimizer and CMB
   reference-value type schemas are structurally closed, subject to their live
   board lanes.
5. Workstation evidence already owed: generator replay/worker invariance,
   live scalar fine-tune parity, artifact save/forge/rebuild, and the full
   40-gate board.
6. Continue the in-file teaching campaign in the priority order above,
   including `compute_data_vectors/` and `cobaya_theory/`.  The bounded README
   stale-fact repair is closed.  Documentation-only batches carry AST-identity,
   compile, and untruncated stale-pattern evidence.
7. Submit BLOAT-01 through BLOAT-03 to the Fable Architect.  If adjudicated
   in, execute them only after active correctness campaigns stabilize their
   contracts.  BLOAT-02 requires a design proposal before code because it
   touches generator checkpoint and append behavior.

Do not mark any item closed from a source-text census alone when the claimed
behavior has an executable public boundary.  The landing handoff must name
the real path run, the mutation that the gate catches, and any capability lane
that remains unavailable.
