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

## Close update after Architect adjudication and manuscript review

This section is the current status.  It supersedes queue states above without
deleting the evidence that led to them.

The preflight finding `RT-IMPL-02` was real, but its first mechanism diagnosis
above was wrong.  The pathspec includes the intended executable roots.  The
actual defect was the shared Git helper's global `stdout.strip()`: stripping
the entire porcelain output removes the first status column when
`gates/board_config.json` is the first line, so `_dirty_lines` cannot recognize
and exclude that line.  Queue 1c-bis repaired that parser and landed as
`279409c`.  The Architect independently audited the real helper in both
directions, including the stripped-first-line mutation arm.  BLOAT-01 landed
in the same commit, leaving one finite-real predicate owner in
`emulator/training.py`.

The staging-order reopen is closed on the code side.  The resident and
disk-backed representations now preserve one selected-row order, and the
banner and boundary cases are covered.  Workstation certification remains a
separate obligation; a Mac/CPU source audit does not promote the workstation
lane to green.  Raw-log deletion, truncation, and byte-edit handling are also
closed on the CPU path.  The executable/input manifest is only partly landed:
phases 1--2 are on main, while phase 3 and its final static-import
reconciliation remain open.  The manifest documentation must continue to name
its inability
to discover dynamic imports and subprocess targets assembled from nonliteral
strings.  Acceptance attacks a driver invoked through a variable, a dynamic
import inside a newly declared root, and a waiver-table entry whose original
site has disappeared.

### Newly adjudicated correctness findings

The Architect independently confirmed all five findings in
`RT-2026-07-13-02..06`; none was reduced in severity or scope.  They are
implementation specifications, not permission for the Red Team to edit the
library.

1. **RT-02, returned-array ownership (unit 66).**  Public predictor and
   diagnostic mappings can expose NumPy views of tensors retained by the
   geometry or a derived cache.  On CPU, `.detach().cpu().numpy()` can share
   storage; on MPS, the device transfer copies.  Caller mutation can therefore
   change future CPU predictions while the same misuse appears harmless on an
   accelerator.  The gate must mutate a returned direct axis and a view of a
   derived cached tensor, repeat prediction, and prove behavioral isolation on
   every supported device.
2. **RT-03, subprocess root identity (queue 1d).**  The board can resolve and
   print project root B while a child process inherits an ambient `$ROOTDIR`
   naming root A.  Inject the resolved root in the child environment at process
   creation, before a check or driver resolves any path.  Red legs include a
   child that rereads `$ROOTDIR` after changing its own environment and a check
   that otherwise resolves data before the injection point.
3. **RT-04, warning-plus-crash false green (queue-1d rider).**  The
   `head-activation-pin` warning leg records the subprocess return code but
   judges only whether warning text appeared.  A process that prints the
   warning and then crashes can pass that leg.  One verdict must require both
   the expected warning and return code zero; a warning-then-nonzero mutation
   must turn red.
4. **RT-05, curved-distance mislabel (blocking unit 67, critical).**  A fixed
   nonzero global curvature can bypass the sampled-parameter check.  The
   background producer labels radial comoving distance $\chi$ as transverse
   comoving distance $D_M$, although curved geometry requires the sine or
   hyperbolic-sine mapping.  At $H_0=70\,{\rm km\,s^{-1}\,Mpc^{-1}}$,
   $\Omega_k=0.1$, and $z=1100$, the independently reproduced values are
   $\chi=13296.826\,{\rm Mpc}$ and $D_M=15538.408\,{\rm Mpc}$, a 16.858 percent
   difference; both are finite and smooth.  Until curvature-capable artifacts
   exist, generation and consumption require a persisted flat-only fact and
   refuse nonzero fixed or sampled curvature before provider work.
5. **RT-06, optional Cobaya display label (unit 68).**  A component without a
   LaTeX display label can fail late even though its scientific name is valid.
   The accepted contract treats the component name as the plain-text fallback;
   a LaTeX label remains optional presentation metadata.

### Twenty-pass teaching and compression review of the manuscript

The 20-pass main-text review used a second-year physics undergraduate who has
not used PyTorch as the reader model.  Each pass looked for one failure class:
undefined dimensions; undefined physics symbols; shape changes hidden inside
prose; views confused with copies; CPU/device movement; mutable defaults;
anonymous callback arguments; unexplained framework lifecycle; validation
after expensive work; residual-block dimension changes; PCE vocabulary;
activation ownership; loss-coordinate changes; family axis conventions;
artifact facts versus code defaults; diagnostic truth sources; gate
catch-power; stale current-state claims; compressed transitions; and generic
machine-written connective prose.  The concrete repairs are in
`texnotes/emulator_code_guide.tex`:

- the first symbol table now defines $N$, $B$, $P$, $P_{\rm enc}$, $D$, $K$,
  and $T_{\rm tmpl}$ in its caption;
- the running example follows four named cosmologies through row selection,
  local coordinates, parameter encoding, target masking, batching, network
  output, decoding, and the physical quadratic error;
- Python mechanics define iterator versus list, eager versus lazy work, views
  versus copies, broadcasting, reshape, dtype, device transfer, factory
  callables, `ModuleList`, classmethods, and mutable defaults before relying on
  them;
- the validation section separates type, finiteness, domain, cross-field, and
  file-dependent checks and shows why quoted YAML booleans and NaN comparisons
  are dangerous;
- a residual block is defined only as the width-preserving map from
  $(B,W)$ to $(B,W)$; the rectangular input and output projections are outside
  that definition, and the skip addition follows the second internal dense
  layer;
- the PCE section defines the polynomial basis, hyperbolic truncation,
  interaction rank, greedy selection, ordinary least squares, PRESS/LOO score,
  all public controls, and a numerical term-by-term example;
- the activation section defines every implemented family, its learned scalar
  count, local derivative, bulk behavior, tail behavior, and head-safety
  question, with a regenerated vector figure whose labels no longer cover the
  curves;
- the family sections state exact CMB, TATT, background, and matter-power axis
  orders and distinguish persisted artifact facts from values inferred from
  current code; and
- the long gate appendix and the file-by-file study route now state what to
  read, what object owns each operation, what to write down, and which
  executable observation is the first evidence.

The anti-template pass removed unsupported praise, vague claims such as
“robust,” canned conclusion sentences, and audit codes from teaching prose.
Current limitations remain labeled as current gaps so they can be deleted
cleanly when their implementation units land.

### PDF custody and freshness ruling

Keep `texnotes/emulator_code_guide.pdf` tracked.  It is a reader-facing product
that can be opened without a TeX installation, and the user explicitly asked
to see the PDF.  Until a documentation gate exists, every change to the TeX or
figure inputs requires a clean two-pass build and visual inspection in the
same documentation commit.  The eventual gate should rebuild in a clean
temporary directory and compare the resulting PDF to the tracked product; it
must not compare timestamps alone.  The current close built 85 pages twice,
with no overfull boxes, undefined references, or cross-reference rerun warning,
then visually inspected the frontispiece, numerical example, validation
section, PCE/design material, activation figure, curved-distance example, and
file-by-file route.

No private writing-guidance material is part of this repository.  No library,
gate, generator, or adapter code was changed by the Red Team in this
documentation close.

## Durable DIDACTICS handoff register: 42--71

This section is the durable copy of the Red Team handoffs that followed the
first DIDACTICS batches.  Findings 1--41 and their Architect adjudications are
already recorded in `gates-and-board.md` and
`state-2026-07-11-and-next.md`.  Findings 42--58 were independently confirmed
by the Architect on her branch while this register was being assembled;
findings 59--71 await her adjudication.  A chat copy is a convenience only:
this file is the source to cite in an Implementer handoff.

The repair standard remains the same throughout: documentation must be
accurate for a second-year physics undergraduate who is new to Python,
PyTorch, Cobaya, and emulators; a gate must execute the observation named in
its title; and a passing assertion must have a non-vacuous, independently
owned reference.

### DIDACTICS-42 -- the package introduction still says cosmic shear only

`emulator/__init__.py` describes the package as a cosmic-shear emulator even
though scalar, CMB, background-grid, and matter-power-grid families are public
and board-listed.  Replace the family-specific headline with the package-wide
role and give a one-line map from each family to its geometry/loss owner.  A
mechanical family census must keep the package index synchronized with the
public modules.

### DIDACTICS-43 -- scheduling prose overstates device and worker guarantees

`emulator/scheduling.py` says `set_device` changes every later default-device
operation and implies worker output cannot interleave.  The first claim is
false for constructors that receive an explicit device or retain an existing
tensor; the second is false for concurrent process streams.  Teach the exact
scope of PyTorch's default device, explicit `.to(device)` movement, and
process-local versus merged output.  Do not promise ordering the program does
not enforce.

### DIDACTICS-44 -- CMB whitening is described as equal physical weighting

`emulator/geometries/cmb.py` and the CMB loss prose conflate raw physical
spectra, covariance-whitened coordinates, and amplitude-law-scaled whitened
coordinates.  Whitening decorrelates and normalizes covariance scale; it does
not guarantee that every training target has sample variance one or that all
physical multipoles receive equal weight.  The repair must trace one
multipole through encode, network residual, law removal, and the physical
quadratic form, naming units and shape at each step.

### DIDACTICS-45 -- `_analytic_R` promises one dtype behavior for two APIs

`emulator/analytics.py` claims one dtype rule for NumPy and Torch.  The Torch
branch preserves tensor dtype and device, whereas the NumPy branch's array
construction and arithmetic can promote to float64.  State both contracts
separately and include float32/float64 controls for each backend.  A prose
claim about dtype is not accepted without an executable dtype assertion.

### DIDACTICS-46 -- selected-log equality can pass with no evidence or failed children

`gates/checks/logscan.py::byte_identity` returns true when both selected-line
lists are empty.  `gates/board.py::_golden_leg` also discards both subprocess
return codes before comparing their selected text.  Thus two empty logs, two
matching lines followed by two crashes, or a one-sided crash after the last
selected line can pass.  Identity requires both return codes to be zero, an
explicit nonempty/minimum selected-line count, and equality of the selected
lines.  Report both return codes and counts.  Mutations for empty selection,
both children exiting one after a matching line, and only the tip child
exiting one must all turn red.  Rename the helper if it compares normalized
text rather than bytes.

### DIDACTICS-47 -- MPS smoke deletes failed rows before testing the dump

`gates/checks/mps_smoke.py` filters zero rows before applying its positivity
test.  An all-zero dump therefore creates an empty selection whose `.all()`
is true; a dump with one zero row can silently discard that row and pass on
the remainder.  Validate the complete stored payload before filtering,
require the expected row count and identity, and distinguish a physically
valid zero from an absent/failed row through the generator's publication
contract.  All-zero and one-zero-row mutations must fail.

### DIDACTICS-48 -- `production-diagnostic` never checks that its PDF exists

`gates/board.py` says the production-diagnostic harness confirms a PDF, but
the executable predicates never assert that a PDF was created, is nonempty,
or is readable.  Either narrow the gate to the outputs it checks or verify the
declared PDF path, nonzero size, parseability, and the expected page/figure
content.  A successful driver that omits the PDF must fail.

### DIDACTICS-49 -- the central experiment lifecycle teaches false stage and artifact facts

`emulator/experiment.py` says staging builds device loaders even though
staging owns NumPy arrays and row coordinates and loader construction happens
later.  It also reverses the `.h5` recipe/geometry file and `.emul` weight
file, and still describes the multi-family experiment as cosmic-shear-only.
Replace the lifecycle with a numbered owner/shape/device trace from raw arrays
through staging, geometry, loader, training, and the two artifact files.

### DIDACTICS-50 -- factored parameter geometry does not increase input width

The parameter-geometry legend says factoring increases network input width.
For the implemented factorization, the layout changes while the total encoded
width remains `n_param`.  Show the unfactored and factored coordinate lists
for one small example and state which operation changes arrangement versus
dimension.

### DIDACTICS-51 -- public training/artifact docstrings omit live arguments and returns

The `run_emulator`, save, and rebuild docstrings lag their signatures and
returned structures: anchor/refine and resolved-training inputs are omitted,
the sixth training return is not defined, and saved/rebuilt information fields
are incomplete.  Add an AST-based public-API documentation census comparing
signature arguments and named return fields with the docstrings.  Each
argument needs type/shape/device/ownership and every return needs its meaning;
do not satisfy the census with placeholder names alone.

### DIDACTICS-52 -- diagnostics-domain's source census proves neither pattern with `or`

`gates/checks/diagnostics_domain.py` joins two forbidden-pattern absences with
`or`.  If either pattern disappears while the other remains, the assertion is
green although its label says neither exists.  Require both absences, report
each count separately, and add one mutation per forbidden pattern so each can
fail independently.

### DIDACTICS-53 -- CMB smoke calls six blocks certified but checks only `cov_tt`

The gate description names numerical certification of the full covariance
product, but the executed numerical comparison covers only `cov_tt` while
other blocks receive shape/finiteness treatment.  Either state that narrower
contract or compare all declared blocks with an independent known-answer
calculation, including a mutation in every block.  One checked block cannot
stand in for six.

### DIDACTICS-54 -- `audit_devices` misses nested tensor owners

The device audit claims to inspect every executed tensor, but its traversal
does not cover tensors hidden in nested containers/owners and it labels a
possible mixed-device runtime failure as performance-only.  Define the object
graph it traverses, recurse through registered modules plus documented nested
containers, and run a board-listed accelerator leg whose mutation hides a
wrong-device tensor at each supported nesting level.  Report unsupported
objects explicitly rather than silently skipping them.

### DIDACTICS-55 -- diagnostic prose converts association into causation

Several diagnostic docstrings call a correlation or one chosen regression
estimator a causal explanation of model error.  Define exactly what is
computed, its conditioning variables, estimator, undefined cases, and what
conclusion it does *not* support.  Terms such as "drives," "explains," and
"responsible for" require an actual causal design; otherwise use association
language.

### DIDACTICS-56 -- the `state_dict` definition wrongly excludes frozen parameters

The model documentation says PyTorch `state_dict()` contains trainable
parameters.  It actually includes registered parameters whether or not
`requires_grad` is true, plus persistent registered buffers.  Contrast
`state_dict()`, `parameters()`, optimizer parameter groups, and
`requires_grad` with a frozen-parameter example.

### DIDACTICS-57 -- masked decode is not the inverse of full-vector encode

`DataVectorGeometry.decode` is described as the inverse of encode even when a
mask discards coordinates.  The kept-coordinate round trip can be invertible,
but reconstructing the original full vector requires a fill/pin policy and
cannot recover discarded values.  State the domain and codomain of both maps,
show a three-coordinate masked example, and reserve "inverse" for the
restricted kept-coordinate mapping.

### DIDACTICS-58 -- 24 gate scripts rely on unexplained Python execution mechanics

Twenty-four scripts repeat a module-level mutable failure list, helper
functions that append to it, an `if __name__ == "__main__"` guard, and
`sys.exit`.  A C reader cannot infer why append needs no `global`, why each
subprocess receives a fresh list, or how the integer exit status reaches the
board.  Add one canonical gate-script preamble defining module global versus
local binding, in-place list mutation, process isolation, the main guard, and
exit-code propagation; reference it concisely from individual scripts rather
than introducing a clever framework.

### DIDACTICS-59 -- eval-batch-invariance does not observe production evaluation

`gates/checks/ge_c_eval_bs.py` computes a separate gate-side array of per-row
chi-squared values, while production `eval_val` publishes aggregate mean,
median, threshold fractions, and history values.  The copied loop can be
invariant while production batching, aggregation, or row association is
wrong.  Exercise the real `eval_val` with one full batch, equal partitions,
and a ragged final batch; compare every published value to an independent
float64 reference.  Include distinct row scores, a row permutation, and a
mutation that drops/reorders one production batch while leaving the copied
helper unchanged.

### DIDACTICS-60 -- scalar smoke gives an unsupported convergence story and stale baseline

`scalar-smoke` attributes two-epoch convergence to target smoothness and
quotes a mean-only median near 0.455.  Recalculation on the current fixture
gives approximately `0.4401868977`.  Describe two epochs as the bounded smoke
budget, not a causal conclusion.  Recompute and report the exact staged-row
baseline, validation-row count, trained median, and threshold; require the
network to beat both the baseline and threshold.  The 0.3 bar may remain only
with recorded empirical margin.  A mean-only/dead-network mutation must fail.

### DIDACTICS-61 -- two gate comments disagree with their fixtures

`gates/checks/stage_ram.py` calls a two-column target fixture one-wide.
`logscan.decreasing` says it needs a nonempty series although a decrease
requires at least two finite observations.  Correct the fixture shape and
define the mathematical predicate: minimum length two, finite values, and the
chosen strict/non-strict comparison.  Empty, one-value, NaN, and equal-value
controls must have explicit verdicts.

### DIDACTICS-62 -- four board entries advertise behavior but check banners

In `gates/board.py`, `head-scheduler-override` does not parse an LR cut;
`berhu-anneal` and `ema-anneal` do not evaluate their schedules;
`joint-training` uses timing rather than parameter change; and
`relu-tanh-norm` does not inspect loss descent.  Either narrow every title and
`WHAT/WHY/HOW` block to startup-schema evidence or add executable behavior:
phase-tagged LR cadence; schedule values before/at/during/after the ramp;
trunk snapshots with nonzero joint delta and exact-zero frozen control; and
finite loss observations with an explicit descent/dead-network bar.  Constant
schedule, ignored override, frozen joint trunk, and mean-only mutations must
red.  Evidence-map reconciliation cannot supply observations these gates
never make.

### DIDACTICS-63 -- parameter-window cuts use their own banner as truth

The gate accepts any line matching `used <digits> of <digits> cut rows`; it
does not parse the values, derive the physical mask, or compare staged row
identities.  The same weak check appears in production-diagnostic.  Use a
deterministic table, independently compute the expected mask, parse both
integers, require `0 <= used <= total`, compare total/used with eligible and
staged counts, and compare exact row identities.  The tight fixture requires
nontrivial shrinkage.  A mutation printing `used 1 of 1 cut rows` while
staging the wrong rows must fail.

### DIDACTICS-64 -- the compile-mode drift arm patches an unused global

`gates/checks/gsv_bitwise_drift.py` advertises drift tests for both activation
defaults and compile-mode defaults.  The latter patches
`training.DEFAULT_COMPILE_MODE`, but rebuild reads the persisted recipe's
`compile_mode`, and the test calls rebuild with `compile_model=False`.
Therefore the patched global cannot affect the result and cannot prove
compile-mode persistence.  Retain the valid activation-default arm, then
either delete the compile claim or add a CUDA compiled lane that rebuilds with
`compile_model=True`, instruments `torch.compile` to observe the persisted
mode, and fails when rebuild ignores or loses that field.

### DIDACTICS-65 -- triangle shading checks global counts, not panel identity

`gates/checks/gt_b_triangle.py` reduces the figure to positive global counts
of shaded panels and marginal bands plus a color count.  It never maps an
Axes object to its x/y parameter labels or compares the observed shaded-panel
set with the expected coverage table.  One grey fill on the wrong panel plus
one unrelated patch can pass.  Construct the exact expected
`(x_parameter, y_parameter, window)` set, extract axis identities, compare the
exact set/count, identify the `omegamh2` marginal, and check `_CUT_GREY` on
each expected artist.  Moving a correct artist to a wrong axis must red while
global counts remain unchanged.

### DIDACTICS-66 -- parity's masked-zero proof can be vacuous and self-defined

`gates/checks/gct_parity.py` derives `masked` from the same rebuilt geometry
under test and accepts a zero nonzero-count.  If `dest_idx` accidentally
covers every coordinate, the masked selection is empty and passes; if a cut
is lost, the object under test redefines the expected mask.  The fixture must
declare an independent, nonempty expected masked-index set, assert observed
mask identity/count, verify exact zeros there, and compare kept coordinates
between section/full outputs.  Mutations setting every coordinate kept and
deleting one expected masked index must fail.

### DIDACTICS-67 -- `FinetuneSource`'s one-open and attribute claims are false

`emulator/warmstart.py` says the source files are opened once and lists every
public carried attribute.  `load_source` first calls `rebuild_emulator`, which
opens `.h5` and loads `.emul`, then opens `.h5` again to read recipe/resolved
facts; the object is constructed once but the HDF5 file is opened twice.  The
attribute list also omits live `.ia`.  Either intentionally consolidate the
reader or teach why a second metadata pass occurs; distinguish object count
from file-open count and define `ia = nla`, `tatt`, or `None`.  Instrumented
`h5py.File`/`torch.load` counts and a constructor-field census must agree with
the final prose.

### DIDACTICS-68 -- warm-start parity does not finite-check perturbed arms

The baseline finetune and transfer parity tensors pass through
`_require_parity_finite`, but the extra-coordinate perturbation encodes and
outputs are compared without that guard.  A model finite on the baseline and
NaN/Inf only after perturbation is mislabeled as "extra parameters leaked";
`torch.equal` is then asked to interpret nonfinite data.  Name the perturbed
encode and output tensors and apply the shared finite predicate before every
comparison in both finetune and transfer.  Board legs must produce NaN and
Inf only on the perturbation and require the finite-contract error with
quantity/side/row.  Removing either perturbed-arm guard must red.

### DIDACTICS-69 -- whitening is repeatedly called equal learnability

`geometries/output.py`, `diagnostics.py`, `warmstart.py`, `designs/plain.py`,
`designs/ia.py`, and `losses/ia.py` say whitening makes directions equally
hard/easy to fit.  Whitening makes training covariance approximately identity
and equalizes numerical variance; nonlinear cosmology dependence, tails, and
network approximation complexity remain different.  Use one canonical
definition: "decorrelated and unit variance/equal numerical scale; this does
not guarantee equal learnability."  A repo-wide multiline scan must leave no
`equally hard`/`equally easy` claim in `emulator/`.

### DIDACTICS-70 -- save documentation contradicts its executed CPU normalization

`emulator/results.py` first says every saved state tensor moves to CPU, then
says a CUDA-saved state needs the saving GPU visible.  The code executes
`detach().cpu()` for every persisted tensor and loads with
`map_location=device`; the original accelerator is not required.  Distinguish
a raw CUDA checkpoint from this library's CPU-normalized checkpoint and teach
that `map_location` selects the load destination.  Assert serialized tensors
are CPU tensors in the save/rebuild gate or cite the existing equivalent leg.

### DIDACTICS-71 -- warm-start's headline promises the bit equality it rejects

`emulator/warmstart.py` says zero-padding reproduces the source function "bit
for bit," while `_PARITY_TOL` explicitly documents numerical, not bitwise,
agreement because different matrix widths change floating-point reduction
order.  State the two distinct invariants: old versus widened networks agree
within `_PARITY_TOL`; rerunning the *same widened network* after changing only
zero-connected extras is bit-identical.  Do not weaken transfer's separate
bitwise base-composition requirement.  Scan every `bit for bit`, `bitwise`,
and `bit-identical` occurrence and map it to the comparator actually used.

## README-focused DIDACTICS handoff register: 72--92

This register is the durable result of a repeated README-only audit at
`7623756`. The review read all four README files (`README.md`,
`emulator/README.md`, `gates/README.md`, and `syren/README.md`) in separate
passes for entry-point orientation, first-time-ML vocabulary, shapes and
numerical examples, command-to-output truth, executable-evidence truth,
cross-file consistency, and stale current-state claims. The target reader is
a second-year physics undergraduate who knows cosmology but is new to Python,
PyTorch, Cobaya, and emulators. The 85-page TeX guide is the depth reference;
the README repair must link to it and condense its essential bridges rather
than copy the manuscript into another long document.

These identifiers are Red Team finding labels, not implementation queue
numbers. The Architect owns adjudication and placement. Where an item extends
an existing queue-6 or scientific-unit contract, that relationship is named
explicitly so the Implementer does not create a duplicate mechanism.

### DIDACTICS-72 -- the README set has no audience route to the long teaching guide

The root README sends the reader only to the package code map
(`README.md:81-82`); it never links the TeX/PDF guide or the gate guide.
`emulator/README.md:18-118` begins with a directory inventory, then jumps to a
module table, without showing the executable order or a recommended study
route. Add a compact **Choose your route** table near the root introduction:
operational configuration in the root README; internals in
`emulator/README.md`; full first-time Python/ML/code-study treatment in
`texnotes/emulator_code_guide.pdf` (with the `.tex` source beside it); and
executable evidence in `gates/README.md`. The package README then needs one
small ownership flow and a nonalphabetical reading order:
configuration -> experiment -> staging -> geometries -> loss -> model ->
batching/training -> results/inference -> adapter. Link to the manuscript for
the full file-by-file route instead of duplicating it.

### DIDACTICS-73 -- setup names dependencies but supplies no reproducible preflight

`README.md:153-171` names CoCoA/CosmoLike, NumPy, SciPy, HDF5, PyTorch, and
CUDA, then immediately defines a relative shell variable. It gives no
supported execution context, environment activation rule, `$ROOTDIR` check,
dependency source, selected-device readback, or zero-training command that
distinguishes an environment failure from a data/YAML failure. Add a short
prerequisite table and one copyable preflight sequence: define `$ROOTDIR`,
state whether this is a CoCoA-embedded rather than standalone install, import
the actual runtime dependencies, print the selected device, and run the
board's list/check-only path without GPU training. State which steps are
CPU-safe and which require the Cocoa + CosmoLike/CAMB workstation.

### DIDACTICS-74 -- commands are not mapped consistently to their products

The single-run example explains the `.emul`/`.h5` pair
(`README.md:173-203`), but the learning-curve, tuning, and bake-off commands do
not all name their output files, table columns, authoritative best recipe,
resume state, or success criterion (`README.md:264-344`). The family driver
table at `README.md:475-505` names wrappers but gives no compact per-family
command pattern. Add one driver-output matrix with: unit of work, command
pattern, shipped template, files and location, table columns, resumable state,
and what a successful run means. Give one row per physical family without
copying five full command blocks. A reader must be able to follow
command -> terminal verdict -> output file -> next consumer.

### DIDACTICS-75 -- symbols and one-row mechanics remain implicit

The six-term glossary at `README.md:562-573` has no shape notation. Later
sections introduce `B`, `N`, `X`, `t`, `n_param`, and `n_dv` piecemeal, while
`C` means the source dictionary's parameter matrix at
`README.md:2462-2468` and covariance at `README.md:673-675,2694-2708`. Add
the manuscript's compact symbol table: `N` complete rows, `B` minibatch rows,
`P` physical parameters, `P_enc` encoded input width, `D` full output width,
`K` kept/predicted coordinates, plus optional template and family axes. Use
`Sigma` consistently for covariance and call `source["C"]` the historical
Python key for the parameter matrix. Follow the legend with one small
numerical row: select matching parameter/target rows, keep a few target
coordinates, encode them, form a residual, and calculate a diagonal
`Delta chi2`. The long four-row example stays in the guide.

### DIDACTICS-76 -- the data boundary is underdefined and described incorrectly

Three related claims teach the wrong storage/transformation model.
`README.md:588-589` sends readers to generation for how parameters are
“sampled, whitened, and named,” although the generator writes physical values
and `ParamGeometry` whitens after row selection. `README.md:573` says the
whole dump is memory-mapped, while `data_staging.py:649-651` memory-maps the
data-vector `.npy` but eagerly parses the parameter `.txt`; the appendix shows
that split at `README.md:2473-2476`. `emulator/README.md:22,128,284-290`
calls the source dictionary “in-memory” even when its resident container
points to a file-backed `np.memmap`. Repair these with one file/source
contract: shapes and reserved columns; exact row-identity equality; physical
values on disk; training-only statistics fitted after selection; keys `C`,
`dv`, `idx`, `dump_rows`, and optional means; and each value's shape,
RAM/file-backed status, and coordinate system. Define the resident container
separately from its possibly lazy numerical payload.

### DIDACTICS-77 -- the controls are introduced before one training step is taught

`README.md:645-658` introduces epoch, minibatch, clipping, and rewind;
`README.md:671-701` relies on gradient votes, trimming, and focal weights; and
`README.md:726-751` introduces AdamW, learning rate, warmup, scheduler, fused
kernels, and validation median. The later pipeline compresses training to one
paragraph (`README.md:2667-2673`). Insert one compact “one minibatch, then one
epoch” table before the loss controls: gather `B` rows -> forward
`(B,P_enc)` to `(B,K)` -> compute scores `(B,)` -> reduce one scalar loss ->
backpropagate -> finite-check gradients -> clip -> optimizer step; after all
minibatches, evaluate held-out validation rows, snapshot the best model, and
step the scheduler. Define gradient, optimizer, learning rate, scheduler,
minibatch, epoch, training set, and validation set. State that validation
performs no parameter update.

### DIDACTICS-78 -- the PCE chapter names a Legendre basis but illustrates monomials

`README.md:1518-1528` uses NPCE, SVD, PRESS, LOO, SGD, mode, least squares, and
greedy selection before defining them. It correctly identifies products of
Legendre polynomials at `README.md:1533-1577`, but the pruning table labels
the terms as raw monomials `x_1^4`, `x_1^2 x_2^2`, and
`x_1 x_2 x_3 x_4` (`README.md:1588-1592`). The implemented basis uses
`P_4(x_1)`, `P_2(x_1)P_2(x_2)`, and products of first-degree Legendre
factors. Expand every acronym at first use; define mode, design matrix,
least-squares coefficient, greedy addition, and leave-one-out error before the
workflow; replace table labels with Legendre products or multi-index vectors;
and show the hyperbolic score for one two-parameter candidate.

### DIDACTICS-79 -- the canonical generator command is not executable and the process model is unexplained

The generator chapter opens with “MPI + emcee + cobaya” and later adds ranks,
workers, differential-evolution moves, autocorrelation, checkpoints, and
failure masks without defining the objects (`README.md:2307-2313,2375-2382`).
Its copyable command has three concrete truth defects: it omits required
`--seed` (`generator_core.py:157-166`), calls `--unif 0` “the default” although
the parser requires the option (`generator_core.py:125-130`), and passes
`--boundary 1.0` (`README.md:2427`) immediately after telling readers only
`0 < boundary < 1` is valid (`README.md:2388-2389`); the code maps that
endpoint to 1 (`generator_core.py:240-242`). Repair the command first: include
and explain a reproducible integer seed, call `--unif` required with its 0/1
meanings, and either omit the boundary for full support or choose and explain
an interior value. Then link one minimal generator YAML per family, define
generator-only keys, annotate which tokens belong to `mpirun` versus Python,
and add a small rank-0/worker/checkpoint diagram. Name what can resume and
which file set constitutes a complete result.

### DIDACTICS-80 -- the direct-Python appendix uses unexplained compact Python

The first scripting example (`README.md:2999-3016`) combines imports, mutates
`sys.path` with an unexplained positional `0`, reads an environment variable
inside nested path construction, and passes constructor arguments in a
compact expression. The profile/background examples then use dense
dictionaries and multi-assignment (`README.md:3039-3069`). Rewrite the
teaching path in the user's code voice: one import per line; named
`repo_path`, `artifact_root`, and `device` variables; explain that
`sys.path.insert(0, path)` fills the first parameter with zero so the
repository is searched first; one constructor argument per line; a named
parameter dictionary built before `predict`; and returned type, shape, units,
and device stated after the call. Begin with CPU and identify the single value
changed for CUDA/MPS. Compact expert variants may follow, but not lead.

### DIDACTICS-81 -- five families share a score interface, not one chi-square metric

`emulator/README.md:97-100` says every family exposes “the same per-sample
chi2.” The common invariant is one score per batch row, shape `(B,)`; the
physical metrics differ. Cosmic shear contracts a physical residual with a
dense inverse covariance; CMB uses persisted per-multipole error scales and
its target law; scalar/grid/grid2d sum standardized residuals. Replace “same
chi2” with “same per-row score interface,” add the actual weighting/formula to
the five-family table, and explain that a common reducer can trim or focus
these scores without making one threshold scientifically equivalent across
families.

### DIDACTICS-82 -- the architecture map conflates bases, trunks, and learned layers

`emulator/README.md:110,260,374` calls NPCE respectively a trunk, a polynomial
base plus neural refiner, and a loss-owned component that is not an SGD
architecture. Use one description: fit and freeze a finite Legendre-polynomial
base, then train a neural residual/refiner. The same map uses MLP, residual,
CNN, transformer, correction head, template, IA, PCE, and SGD before defining
them (`emulator/README.md:139-145,252-269,356-386`). Add a design table with
input/output shape, fixed part, trained part, and intended use; define a
residual block as same-width `h -> h + F(h)` and distinguish it from
rectangular entry/exit projections. Narrow the root README's stronger claims:
`README.md:929-980` says every learned layer is linear and the two trunk
projections are the only rectangular learned matrices, but Conv1d, attention,
and optional `FiLMGenerator` (`blocks.py:350-425`) are counterexamples. The
square-dense/two-projection rule belongs to the ResMLP trunk, not the library.

### DIDACTICS-83 -- dependency and physics-owner summaries overclaim

`emulator/README.md:89-91` observes that only the cosmic-shear geometry imports
CosmoLike, then incorrectly concludes that every other path is “pure
PyTorch.” Those paths also use NumPy, SciPy, YAML, psutil, HDF5, and optional
Matplotlib/GetDist; generators and smokes may require Cobaya/CAMB. State the
precise claim: non-cosmic-shear training does not import CosmoLike, then list
or link common and optional dependencies. `emulator/README.md:115-118`
separately says both physics modules are shared by generator, adapter, and
gates. That is true of the Syren base path; the background generator asks
CAMB directly, while `emulator/background.py` owns consumer-side distance
integration for adapter, diagnostics, and gates. Split the rows and name the
actual consumers.

### DIDACTICS-84 -- the function census promises detail it does not contain

`emulator/README.md:277-280` directs readers to docstrings “for full detail,”
then provides symbol lists that omit inputs, outputs, shape, dtype, device,
copy/view/lazy behavior, mutation, and next caller. This is most damaging for
staging (`:282-292`), training (`:396-409`), experiment (`:411-422`),
persistence/inference (`:440-448`), and scheduling (`:434-438`). Keep leaf
modules as an index, but give those lifecycle owners a compact table: public
entry, input and shape, returned or mutated object and shape, storage/device/
dtype, eager/lazy or copy/view behavior, and next consumer. Call section 6 an
index and link the manuscript's file-by-file route. This does not replace the
queued in-file docstring campaign; it prevents a false claim that the detail
already exists.

### DIDACTICS-85 -- the advertised new-family recipe ends before persistence and service

The “NEW output family” row at `emulator/README.md:231` lists a geometry, loss,
experiment branches, predictor, adapter, and two gates, but omits source
generation or an external-source contract; geometry `state`/`from_state`;
coordinate/head capability; family facts in save/rebuild/readback; driver and
example YAML; diagnostics/plotting; and documentation. Replace the long cell
with a lifecycle checklist: schema/validation, source and sidecars, geometry,
score, model-coordinate capability, experiment, artifact facts and rebuild,
predictor, adapter, driver/template, diagnostics, identity gate, smoke gate,
and docs. Each row names input/output shape and permits “not applicable” only
with a reason. A family that trains but cannot rebuild or label its physical
output is not integrated.

### DIDACTICS-86 -- stable code maps are mixed with contradictory audit chronology

The package map embeds long temporary defect reports, dates, run anecdotes,
and decision shorthand inside stable role cells
(`emulator/README.md:128-130,155,164-169,179-216,265-273,419`). This is the
already-queued README/current-limitations restructuring, now with two factual
catch-power examples: line 128 says grid2d law staging “currently defeats” the
memory ladder while line 419 says bounded streaming/spill landed and the
materialization issue is retired; line 83 names the old
`external_modules/code/emulators/emultrfv2/` path while this repository uses
`external_modules/code/emulators_code_v2`. Queue 6 must leave one short
present-tense role sentence per map row, one separate dated Current
limitations table with consequence/status/descriptive note link, and no
board-run anecdote or internal ruling language. Note references must be
clickable paths rooted from `emulator/README.md`, for example
`../notes/<file>.md#<stable-anchor>`, and a link check must cover all READMEs.

### DIDACTICS-87 -- the gate README teaches obsolete scope, resume, and log semantics

`gates/README.md:3-7` calls the harness a cosmic-shear suite that runs “every”
test and writes one log per test. The board covers five families; default
selection excludes optional gates, current PASS gates may resume without
execution, and unmet dependencies can skip a selected body. The file promises
`logs/<test>.log` (`gates/README.md:46,147`), while
`run_board.py:1895-1937` writes immutable timestamped per-attempt logs through
an atomic `.inprogress` publication and stores the exact filename/digest in
status. Rewrite the opening as the command-line acceptance harness for all
five families. Define default selection, optional gates, and “success = every
selected gate is current PASS.” Teach the resume states PASS, FAIL,
stale-code, stale-input, stale-log, pre-manifest, interrupted, SKIP-DEP, and
not-run; only a PASS with current code, input/manifest, and raw-log identity
may be reused. Readers must follow `BOARD.md` or `board_status.json` to the
cited attempt rather than guess a filename.

### DIDACTICS-88 -- the gate table omits foundational gates and assumes its evidence vocabulary

`gates/README.md:101-108` admits its central table is incomplete. The current
board has 40 gates, while the table omits `finite-contract`,
`board-selftest`, `generator-seed`, `cli-strict`, `family-first`, `stage-ram`,
`artifact-readback`, and `diagnostics-domain`. Its rows then use EMA, BerHu,
NPCE, parity, round trip, bitwise, collapse bar, known answer, fixture, census,
sidecar, CAMB, and Cobaya without a local primer. Complete or mechanically
check the table from board metadata. For each gate give plain-language claim,
production path or fixture, actual verdict, catch-power mutation/control,
prerequisites, and what it does *not* prove. Precede it with the manuscript's
reduced evidence ladder: banner/schema observation < parity/round trip <
independent known answer with mutation catch power. This is distinct from
DIDACTICS-62: evidence-map rollout cannot manufacture observations a gate
never executes, and the README must not relabel those weak gates as stronger.

### DIDACTICS-89 -- board operator instructions contradict portable configuration and preflight

`gates/README.md:66-72` tells every user to edit `board_config.json`; its own
`_help` says a standard clone resolves `rootdir: null` from `$ROOTDIR` and
needs no edit. The README also says preflight enforces a “stale git tip”
(`:70,83-84`), but executable preflight proves the embedded base-notes commit
is an ancestor of HEAD; it does not fetch or compare `origin`. Replace the
edit instruction with: export and define `$ROOTDIR`, inspect the portable
config, override only a nonstandard deployment, then run `--check`. Add
examples for `--list`, one named gate, the optional triangle gate, one tier,
`--from`, `--force-rerun`, and `--force-rerun-all`; state exit 0 (selected
surface current green), 1 (preflight/gate/dependency non-green), and 2
(usage/board-authoring/manifest error). Distinguish terminal summary,
`BOARD.md`, status database, and immutable raw attempt log.

### DIDACTICS-90 -- the Syren README universalizes one of three target laws

`syren/README.md:3-7` says the MPS network learns `log(P/P_base)` and
multiplies the formula back. The registry has three paths: `none` learns raw
rows; `syren_linear` learns `log(P/P_base)`; `syren_halofit` learns
`log(B/B_base)`, with `B=P_nonlinear/P_linear`. Add a three-row table with
quantity, axes, units, training target, and reconstruction. Define `P(k,z)`,
linear/nonlinear power, boost, `k`, `z`, `h`, and units; call Syren a
deterministic analytic approximation/fit, not “the exact formula.” Exactly
executing reconstruction does not make the approximation physically exact.
Include one number, for example `P_base=900`, `P=990`,
`c=log(990/900)`, and `900 exp(c)=990`, then extend to a z-outer
`(N_z,N_k)` surface and diagram raw+base -> staged ratio -> correction ->
adapter base -> physical product.

### DIDACTICS-91 -- Syren provenance promises identity the artifact does not store

`syren/README.md:5-7,46-50` says vendoring means artifacts and bases can never
drift and that artifacts record the base by construction. Current
`Grid2DGeometry.state()` persists law name and geometry, not a Syren
implementation digest; the root README admits artifacts do not yet bind
implementation identity (`README.md:197-198`). This rides the existing
implementation-identity unit: narrow current prose to “vendoring prevents an
unreviewed package-manager upgrade”; until the manifest lands, a repository
revision can change inference formulas and producing-commit identity/retraining
remains necessary. `syren/README.md:34-35` also calls bodies “byte-verbatim
(AST-verified).” An AST comparison discards comments and formatting and cannot
prove byte identity, and no retained vendoring probe is present. Either retain
a source snapshot plus clearly named normalized-AST comparison, or state which
source was copied and which import/header edits were made. Reserve
“byte-verbatim” for an actual byte/hash comparison.

### DIDACTICS-92 -- the Syren README has no user-facing domain or evidence route

The 50-line file lists function names and paper identifiers but gives no fit
domain, units, callable contract, use/verification command, or evidence
limits. Add descriptive paper titles/roles and a compact function table with
inputs, output, shape, units, and accepted domain; this domain paragraph rides
the existing MPS interpolator/domain unit rather than inventing ranges. Add
commands for board-listed `mps-identity` and `mps-smoke` with prerequisites
and an honest evidence split: `mps-identity` uses controlled/stub bases to
prove assembly algebra, while the current `mps-smoke` real-CAMB path exercises
law `none`; neither alone proves numerical accuracy of the real Syren fit.
State what is tested, what remains outside the gates, and when re-vendoring
requires retraining.

### DIDACTICS-93 -- a covariance loop calls its numerical step `h`, colliding with cosmological `h`

`compute_data_vectors/compute_cmb_covariance.py:472,558-573` uses `h` for a
finite-difference step fraction in prose, the loop variable, and the
`stencil_derivative` call. In cosmology, `h` already means the dimensionless
Hubble parameter `H0/100`; for the repository's Planck-LCDM fiducial it is
`0.6736`. The collision caused a real audit misunderstanding and therefore is
not merely cosmetic. Rename the numerical value to `step_frac` in Python and
`s_step` in explanatory equations, including derivative helper parameters,
diagnostics, and persisted labels. Do not change arithmetic or artifact
schema values. The AST-minus-docstrings acceptance must show only identifier
and documentation changes, and the Planck-LCDM covariance control remains
byte-identical.

The same completion pass must use an untruncated `h` census, not stop at the
first file. Current additional collisions include EMA horizon
(`training.py:1353`), neural hidden states (`designs/plain.py`,
`designs/ia.py`, and `designs/blocks.py`), focal hardness
(`losses/core.py:612-623`), and local Hubble-function arrays
(`cobaya_theory/emul_baosn.py:333`, `gates/checks/bsn_smoke.py:374`). Use
names such as `horizon_epochs`, `hidden`, `hardness`, and `hubble_values`.
This is a readability rename, not permission to change equations or persisted
field names.

### DIDACTICS-94 -- fine-tuning names an anchor without teaching the L2 weight-displacement contract

`README.md:1748-1755` calls the planned fine-tune anchor only “a pull back
toward the saved weights.”  A first-time machine-learning reader is not told
what is measured, what `anchor` controls, how this differs from weight decay,
or which new-input weights must remain free.  The user's requested explanation
is now specified in `artifacts-inference-warmstart.md`, “README teaching
rider”: define the conceptual L2-SP displacement penalty, then show the actual
decoupled post-optimizer update, with every symbol, one scalar numerical
example, the new-column mask, and the current public refusal.

Accuracy constraint: do not say CoCoA SONIC currently adds this penalty to
the scientific loss.  `emulator/training.py:323-385` deliberately keeps the
anchor outside the loss and applies
`W <- W - lr * lambda * mask * (W - W_0)` in place after
`optimizer.step()`, so AdamW's adaptive moments do not rescale the pull.
`emulator/warmstart.py:977-1007` masks the padded columns for newly added
cosmological inputs.  `emulator/warmstart.py:159-165` still refuses a
fine-tune `anchor` at the public validator, so the README must teach the
queued behavior without advertising it as live.  This is a documentation
repair owned by the existing unit-24 anchor-truth campaign, not a second
implementation unit.

### DIDACTICS-95 -- the README set is still a development diary instead of a current library guide

The user's current-state ruling applies to every README. A README may explain
what the code does, how to invoke it, what its files mean, and a present
restriction that changes a user's action. It must not narrate the sequence by
which the implementation arrived there. The root README currently violates
that boundary repeatedly. Representative examples include:

- `README.md:1935-1936`, which inserts a paper's attention-versus-MLP result
  into the CMB usage introduction;
- `README.md:1979-1988`, which reports workstation, board-run, numerical-error,
  fixture, rerun, and evidence-note status;
- `README.md:2009-2014`, which describes a retired amplitude formula, its
  conditioning failure, and its retraining response;
- `README.md:184-203,346-351,470-473,507-516,591-596,1625-1650,
  1748-1757,2225-2232,2279-2301`, which mix queued repairs, gate status,
  rejected or legacy choices, and roadmap state into current usage prose.

The Implementer must perform a README-set current-state pass. Move proof
history, benchmark comparisons, rejected alternatives, retired-formula
rationale, dated decisions, queue position, gate-run details, and repair plans
to their existing note owners or the manuscript. Do not create a new note per
README paragraph. A current limitation may remain only in this form:

1. **Scope:** the precise configuration or interface affected.
2. **Consequence:** what result is unavailable, unsafe, or different.
3. **User action:** what the reader must do now.

No date, discovery story, future landing plan, or acceptance status belongs in
that limitation. Scientific citations may remain when they identify the
source of an implemented equation or vendored scientific fit. A claim that
another architecture performed better belongs in the paper, not in usage
instructions.

Acceptance is an untruncated scan across every tracked README for the complete
history vocabulary, followed by human adjudication rather than blind deletion.
The candidate family includes `board run`, `workstation-proven`, `maximum
relative error`, `fixture`, `rerun`, `queued`, `in flight`, `landed`,
`remaining command`, `ruling`, `adjudicat`, `hard-won`, `retired`, `earlier
raw`, `design record`, and dated status language. Every retained hit needs a
reviewed reason in the audit evidence showing that it describes a present
user contract rather than development chronology; do not insert audit
justifications into the README itself. Commands, paths, anchors, equations,
and YAML examples retain the existing executable/path/anchor acceptance
checks.

### DIDACTICS-96 -- the CMB README section needs a conceptual and factual rewrite

`README.md:1920-2089` is the strongest instance of the diary problem and is
too compressed for a first-time machine-learning reader. It opens with nearly
the entire shared training stack, correction-head mechanics, tokenization,
NPCE restrictions, and a literature result before it has stated the four
files in the workflow. Its covariance, amplitude-law, and roughness passages
also contain claims broader than the implementation.

Rewrite the section in this order:

1. Define one artifact as one spectrum on one stored multipole grid; define
   TT, TE, EE, `pp`, `C_ell`, and multipole `ell` before using them.
2. Show the producer/consumer flow without falsely linearizing it.
   `dataset_generator_cmb.py` writes the spectrum dumps, while the independent
   `compute_cmb_covariance.py` writes the covariance `.npz`. Both products
   feed `cmb_train_emulator.py`; its `.h5` plus `.emul` pair feeds
   `emul_cmb.py`, which provides Cobaya `Cl`. The covariance script does not
   consume the generated spectrum dumps.
3. Explain the target transform and per-multipole scaling, with a minimal real
   YAML block.
4. Name the supported models and warm-start restrictions by pointing to their
   main sections instead of restating the entire training stack.
5. Explain the saved artifact and serving units.
6. End with a short current-limitations list containing only user actions.

Four factual corrections are binding:

- The single formula now shown at `README.md:1965-1969` is the auto-spectrum
  TT/EE form, not a generic variance for every supported spectrum.
  `compute_data_vectors/compute_cmb_covariance.py:229-242` uses a different TE
  expression and a lensing-potential expression without instrumental noise.
  The README need not reproduce every derivation. It can state the executable
  interface: the `.npz` supplies `ell`, `cl_<spectrum>`, and
  `sigma_<spectrum>`, and the trainer divides the centered residual at each
  multipole by the stored `sigma_<spectrum>`.
- Optional non-Gaussian mode writes dense `cov_*` blocks
  (`compute_cmb_covariance.py:732-748`), but the current trainer reads only
  `ell`, `sigma_<spectrum>`, and `cl_<spectrum>`
  (`emulator/experiment.py:3746-3753`). Enabling the dense calculation does
  not presently change training whitening. The README must say so rather than
  imply that every covariance product feeds the loss.
- The `as_exp2tau_ref` target removes the dominant primary-spectrum amplitude
  dependence; it is not “the SHAPE only.” `as_name` identifies a positive
  linear-amplitude column present in the parameter dumps. The generic
  generator writes whatever sampled column names its YAML declares, so the
  README may say the shipped example uses `As`, not that the generator always
  samples `As` directly. The retired `exp(2 tau)/A_s` history and `5e8`
  conditioning story move to the manuscript/note. The law is physically
  motivated for the primary TT/TE/EE spectra. The validator currently also
  permits it for `pp`, so the README must not recommend that combination
  without qualification.
- The roughness term is not a sharp high-pass cutoff that proves physical
  structure “passes untouched.” The implementation applies the same boxcar
  moving average twice, subtracts that triangularly smoothed result from the
  original residual, and penalizes the squared remainder. Define `lam` as the
  added weight and `period_cut` as the smoothing-window scale. State that the
  reported chi-square remains the ordinary residual metric; avoid absolute
  claims about which physics it can never affect.

The serving paragraph must state its convention directly: the adapter returns
raw `C_ell`, not `ell(ell+1)C_ell/(2 pi)`. TT, TE, and EE use `muK2`; `pp` is
dimensionless. Requests outside the stored multipole grid are refused.

Remove the arXiv performance aside, board/run numbers, measured `6e-14`
evidence, fixture/rerun status, gate names, “Epoch 0” proof language, and the
evidence-note pointer from this user section. Those facts remain available in
the manuscript and existing CMB/gate notes. Acceptance includes a code-to-
README key census for `ell`, `cl_*`, `sigma_*`, and dense `cov_*`, plus a
current-output inspection of the covariance script; a prose-only rewrite is
not enough.

### DIDACTICS-97 -- essential explanations are hidden inside parenthetical asides

Parentheses are useful for a symbol, unit, acronym, or one short local
definition. They become a parsing hazard when the reader must hold the main
sentence open while learning an algorithm, exception, comparison, or second
argument. The current root README has 72 prose paragraphs with at least two
opening parentheses after fenced code and table rows are excluded.
`emulator/README.md` has 177 of 482 source lines containing a parenthetical and
79 lines longer than 160 characters. These counts are candidate baselines,
not automatic failure thresholds.

The worst root clusters are the loss and scheduler descriptions
(`README.md:673-708,796-870`), model and phase explanations
(`README.md:929-951,1086-1092,1133-1137,1456-1494`), warm-start discussion
(`README.md:1714-1817`), CMB opening and amplitude law
(`README.md:1920-1940,1992-2022`), pipeline/memory explanation
(`README.md:2434-2673`), and scripting interface
(`README.md:2977-2985`). `emulator/README.md:128,145,162,165,169,205,211,
216,330,419,448` are representative nested-README candidates.

For each candidate, read the sentence without the parenthetical. If the
reader loses a required input, transformation, exception, limitation, or
reason for the next step, promote that material to a complete sentence, a
named table row, or a diagram label. A rewritten paragraph should normally
have one subject and one job. Do not “fix” the count by deleting definitions
or by compressing mechanics into denser Python vocabulary. Preserve short
units, mathematical symbols, acronym expansions, links, and code-call
signatures where parentheses are their natural syntax.

Acceptance records both pre/post candidate counts and a reviewed sample from
every README, but no numerical target substitutes for undergraduate read-
through. A first-time ML reader must be able to identify the owner, input,
operation, output, and next consumer without resolving a nested aside.

### DIDACTICS-98 -- the current-state and novice-order defects extend beyond CMB

The rest of the root README needs the same pass, section by section rather
than one global find-and-replace. Preserve its useful YAML and diagrams. Apply
one family template: (1) physical mapping and symbol definitions; (2)
generator outputs; (3) target transform and scaling; (4) training command and
minimal family YAML; (5) saved output and inference serving; (6) present
limitations only.

- **Scalar (`README.md:1833-1914`):** define the sampled inputs and named
  scalar outputs before “fast” and “slow” jargon. Split the covariance-header,
  output-column, and independent-standardization contract into separate
  sentences. Replace the architecture/fine-tune/transfer/PCE/cuts paragraph
  with an explicit support list: `resmlp` only; fine-tuning requires identical
  output names in identical order; `finetune.anchor` is refused; transfer and
  refinement are unsupported; optional PCE uses `form: residual`. Remove the
  design-record pointer.
- **Expansion history (`README.md:2095-2175`):** do not state a universal
  training grid `z in [0,3]`. The generator requires `0 < z_min < z_max`, the
  shipped example begins at `0.001`, and the adapter extrapolates its `H(z)`
  interpolator down to zero before integrating. Teach the two artifacts and
  their target laws first. Make the flat-universe warning exact: the artifact
  named `D_M` currently stores radial comoving distance `chi`; flat formulas
  then produce `D_A` and `D_L`; users must set `omk: 0`. The loader detects
  curvature only when `omk` is an artifact input. State that present
  scope/consequence/action contract without gate comparisons.
- **Matter power (`README.md:2181-2301`):** define `z`, `k`, `P(k,z)`, the
  linear spectrum, nonlinear boost, units, and the Syren base before the law
  equations. Do not claim low-`k` boost is universally one and therefore
  pinned. `Grid2DGeometry` pins a law-space column when its variation is
  unresolved at the geometry's float32 threshold,
  `scale <= 8 * eps32 * abs(center)`; this is a measured representation rule,
  not a universal physical law. Remove gate closure, workstation evidence,
  acceptance-experiment history, and future landing order. Retain three
  direct current restrictions: MPS PCE is suitable only for small datasets;
  the current `sigma8` helper uses the wrong radius convention; and Cobaya
  product registration plus the `As`/`As_1e9` requirement still block
  production EMUL2 serving.
- **Shared mechanics (`README.md:184-212,346-351,470-516,591-596,
  673-870,929-951,1167-1172,1234-1239,1393-1400,1456-1494,1625-1696,
  1714-1817,2434-2673,2742-2821,2954-2971,3035-3078`):** remove queued
  repairs, rejected alternatives, paper comparisons, internal design debate,
  and migration narrative. Retain the present operation and the action a
  user must take. Put implementation mechanics needed by maintainers in the
  existing code-map README or topic note, not in a parenthetical in the run
  guide.

Every family rewrite must be checked against its validator, generator,
geometry/loss owner, and adapter. A README statement is not accepted merely
because another README already says the same thing.

### DIDACTICS-99 -- the nested READMEs need the same boundary, with different jobs

This is a rider on the existing DIDACTICS-86--92 file visits, not a second
documentation campaign.

- `emulator/README.md` currently combines a conceptual file map, a second
  near-complete function inventory, and repair chronology. Keep one compact
  map and add a novice reading order that follows execution:
  driver/config -> experiment -> staging -> geometries/model/loss -> training
  -> results -> inference -> adapter. `EmulatorExperiment` is the orchestrator
  that invokes those later owners; do not place it after training. Each step
  names its input, output, and why the next owner needs it. Remove “fix
  queued,” “in progress,” dated rulings, superseded designs, gate adequacy,
  and the CMB board-run paragraph at line 216. A present unsafe feature gets
  one current-restrictions entry, not a queue diary.
- `gates/README.md` legitimately explains evidence, but it must explain the
  current acceptance mechanism rather than recount past bugs or runs. Replace
  the nested selector paragraph with an option table. Replace the knowingly
  stale “40 gates” list with generated/current categories and point to
  `run_board.py --list` as authoritative. CMB rows name the high-level
  contracts; exact legs, mutation arms, numerical errors, and workstation-run
  history stay in the gate home note. Define internal terms before the table.
- `syren/README.md` may retain scientific attribution and licensing. Replace
  dated copying history and the unsupported “byte-verbatim (AST-verified)”
  statement with the exact upstream snapshot, the present deviations, and an
  honestly named comparison method. Add plain definitions for `P(k)`, linear
  power, nonlinear boost, input/output shapes, units, and supported domain.

There is currently no README under `compute_data_vectors/` or
`cobaya_theory/`. Do not create two more files merely to make the tree
symmetrical; teach those packages in the root workflow and code map unless a
future standalone user task justifies its own guide. Completion runs the
history-vocabulary and parenthetical candidate scans over every tracked
README and records a reviewed allowlist.

### DIDACTICS-100 -- SONIC has one exact public expansion

The user-defined name is **S**imulated **O**bservables for **N**umerical
**I**nference in **C**osmology. `README.md:6,21-22` and
`texnotes/emulator_code_guide.tex:15,23-24` currently expand SONIC as
“Surrogates and Operators for Numerical Inference in Cosmology.” The root
README correction belongs in the Implementer's README visit. The TeX source
is under Red Team custody and is corrected in this same Red Team landing.

Acceptance is an untruncated, case-sensitive scan of public README and TeX
sources: zero instances of the retired expansion; every spelled-out instance
uses “Simulated Observables for Numerical Inference in Cosmology.” In Markdown
the first letters are bold exactly as shown above. The acronym, filenames, and
artwork path remain `SONIC`; no code or artifact schema changes.

#### Fable adjudication of DIDACTICS-95--100 (2026-07-13)

All six findings are **CONFIRMED**. The user's current-library/not-a-diary
ruling is the binding standard for every README. Its acceptance remains the
candidate scan followed by human adjudication, with reasons recorded in audit
evidence rather than inserted into the README. DIDACTICS-96's code comparison
also established that three of its four CMB overclaims conflict with the
repository's own more accurate loss docstring, geometry preamble, or covariance
implementation: the public teaching prose drifted beyond its owners. The wave
folds into the existing root, emulator, gates, and Syren README visits; it does
not create new queue slots. DIDACTICS-100 keeps the custody split already
executed: the Red Team corrected the TeX source and PDF, and the Implementer
corrects the root README. The DIDACTICS register is now 001--100.

### README audit exclusions and existing owners

The loop re-observed, but did not duplicate, four existing contracts:
DIDACTICS-69 owns “whitening makes every direction equally easy” language;
DIDACTICS-62 owns board rows whose claims exceed banner-only checks; queue 6
owns the stable-role/current-limitations split and removal of internal audit
shorthand; and the evidence rollout owns the two-layer check-script mechanism
plus declared-versus-executed reconciliation. DIDACTICS-86 and 88 sharpen
their README acceptance surfaces; they do not create parallel implementations.

### Durable-record rule

Every future Red Team handoff is appended to an existing topic or audit note
before, or in the same turn as, its chat copy.  The final handoff cites that
file and commit.  Chat history is never the sole copy of an Implementer
contract.

## Durable correctness-handoff registry

This registry applies the durable-record rule retroactively to the earlier
bug hunts. It does not replace the detailed domain notes. It tells a cold
reader which identifier series exist, where the complete contract lives, and
which apparent number gaps were never issued. The census was repeated over
all Markdown files and Git history at `cf95a15`.

### The pre-45M queue: do not confuse three numbering scopes

The historical state note used three scopes. They are made explicit here so
"unit 3" cannot silently mean two different defects.

| Scope | Identifier | Finding and durable owner |
|---|---:|---|
| Original queue | 1 | CMB equation-6 normalization — `families-scalar-cmb.md`. |
| Original queue | 2 | Dataset readiness — `data-generation-and-cuts.md`, "Original unit 2"; MPS sigma8 half — `families-background-mps.md`. |
| Original queue | 3 | Best-record/selected-model truth — superseded by the full selection-record contract in `training-stack.md`. |
| Original queue | 4 | Harness and CLI truth — `gates-and-board.md`. |
| Original queue | 5 | A historical bundle, now decomposed: scalar-driver key ownership — `families-scalar-cmb.md`, "Original unit 5(a)"; NPCE — global 19; optimized assertions — global 12; remaining clauses route to artifact, adapter, and plotting owners. |
| Second-wave *local* list | 1 | Bounded grid2d staging; later canonical global unit 27 — `data-generation-and-cuts.md`. |
| Second-wave *local* list | 2 | Data-selection truth; absorbed into the file-set authenticity work (global 8, 25, 26, and 28) — `data-generation-and-cuts.md`. |
| Second-wave *local* list | 3 | Artifact-pair-integrity campaign alias — `artifacts-inference-warmstart.md`. This is **not** original/global unit 3. |
| Second-wave *local* list | 4 | Parallel completion truth; merged into global unit 10 — `training-stack.md`. |
| Second-wave *local* list | 5 | Live resource sizing and table-length truth — `training-stack.md`, sizing/resource gaps. |
| Second-wave local list / later queue 6 | 6 | Python documentation truth — `conventions-and-workflow.md`. |
| Global queue | 7 | Real Cobaya adapter contract — `artifacts-inference-warmstart.md`. |
| Global queue | 8 | Checkpoint-set integrity — `data-generation-and-cuts.md`. |
| Global queue | 9 | Validation/diagnostic memory and totality — `training-stack.md`. |
| Global queue | 10 | Worker liveness plus completion truth — `training-stack.md`. |
| Global queue | 11 | Geometry numerical/read-side integrity — `artifacts-inference-warmstart.md`. |
| Global queue | 12 | Validation parity under `python -O` — `conventions-and-workflow.md`. |
| Global queue | 13 | CMB covariance-input schema — `families-scalar-cmb.md`. |
| Global queue | 14 | Finite training/evaluation contract — `training-stack.md`. |
| Global queue | 15--16 | BAOSN and MPS domain totality — `families-background-mps.md`. |
| Global queue | 17 | Generator ingress identity — `data-generation-and-cuts.md`. |
| Global queue | 18, 20, 22--23 | Schedule, range, selection-record, and run-control truth — `training-stack.md`. |
| Global queue | 19, 29 | NPCE LOO and model-block value schema — `models-and-designs.md`. |
| Global queue | 21, 24 | Inference numerical boundary and fine-tune anchors — `artifacts-inference-warmstart.md`. |
| Global queue | 25--28 | Nested paths, validation axes, bounded staging, and validation leakage — `data-generation-and-cuts.md`. |

The word "unit" without one of these scopes is historical shorthand, not a
safe identifier. New notes cite the finding name and owner heading as well as
a number.

### 45M finding series

Every issued 45M finding through 90 has a durable record. The canonical
ledger for 01--72 is `state-2026-07-11-and-next.md`; each live contract routes
to its scientific topic note. Findings 73--90 are indexed in
`red-team-implementer-handoff-2026-07-13.md`; their detailed records live in
`gates-and-board.md` and, where applicable, the scientific topic note.

- `45M-10` and `45M-18` were **not issued**. They have no note, repository,
  or Git-history source. The gaps are intentional tombstones; no future
  finding may reuse them.
- `45M-05` was retracted by the Red Team. The surviving evidence says the
  ordinary conversion chains were accepted and no source-style gate was
  required. The original full handoff is not available and must not be
  reconstructed as a live defect.
- `45M-08` was received only as an index item. The independently verified
  portion is complete in `artifacts-inference-warmstart.md`: unconditional
  covariance overwrite plus non-transactional publication, with
  preexistence-refusal and interrupted-write legs. If an original longer
  handoff is recovered, append only genuinely additional clauses and retain
  this provenance limitation.
- `45M-43` and `45M-44` were retracted after validator reachability disproved
  them. Unit 54 is withdrawn and its number retired; `training-stack.md`
  retains the VOID analysis and audit lesson.
- `45M-25` now has its missing domain-owner contract in
  `training-stack.md`, "Scheduler execution protocol".

Bundled implementation summaries are disaggregated as follows:

- `45M-71` = resume input identity plus atomic `RUNNING` state;
- `45M-74` = immutable per-attempt logs plus atomic status/board publication;
- `45M-73` = a dependency-skipped gate executed no body but the command
  returned success;
- `45M-77` = unknown or mixed selectors silently changed or emptied the
  selected gate surface; and
- `45M-82` = an unavailable mandatory `torch.compile` lane returned success.

Current-status qualifications remain part of the record: 72's foundation is
landed but its full evidence rollout is open; 75's schema half is landed while
post-optimizer finiteness is workstation-owed; 76 and 79 retain live-proof
obligations; 81's seed landing was reopened by the demonstrated append/RNG
restart and its current contract lives in `data-generation-and-cuts.md`; 85's
broader residue scan remains; 86 and 90 were reviewed as partial didactic
drafts; and 89's original "two of seven" statement was corrected to exactly
one in-code verdict.

### 20M, RT, and BLOAT series

`20M-01` through `20M-19` and `20M-21` through `20M-25` each have an
Architect adjudication in `gates-and-board.md` and a full domain-owner
contract. `20M-20` was **not issued** and is a retired gap, not missing
content. Owners are:

- background/MPS: 01, 05, 06, 08, 18, 19;
- scalar/CMB: 02, 03, 04, 14, 22;
- artifacts/inference: 07, 09, 10, 11, 17;
- training: 12, 13, 24, 25; and
- generation/publication: 15 (the checkpoint-ingress amendment to unit 56),
  16, 21, 23.

`RT-2026-07-13-01` is recorded in `gates-and-board.md`. The following
canonical labels name the five findings already described in this file:

- `RT-2026-07-13-02`: returned-array ownership;
- `RT-2026-07-13-03`: subprocess root identity;
- `RT-2026-07-13-04`: warning-plus-crash false green;
- `RT-2026-07-13-05`: curved-distance mislabel; and
- `RT-2026-07-13-06`: optional Cobaya display label.

`RT-IMPL-01` through `RT-IMPL-04` have dedicated sections near the beginning
of this file. `BLOAT-01` through `BLOAT-04` are also fully recorded here and
in `gates-and-board.md`; BLOAT-02 and BLOAT-03 additionally have scientific
topic owners. No correctness finding in these issued ranges depends solely on
chat history after this registry.

### 25M finding series (2026-07-13 continuation)

The 25-minute continuation uses a fresh `25M-` namespace so it cannot be
confused with the closed 45M or 20M series. Every issued identifier has a
public-reachability proof, concrete wrong result, required contract, and
discriminating leg set in its scientific owner before any chat handoff:

- `25M-01`: uniform-prior interior is scaled by absolute coordinate and can
  shrink or invert a legal interval — `data-generation-and-cuts.md`;
- `25M-02`: temperature changes uniform support but is absent from output
  identity — `data-generation-and-cuts.md`, unit-8 manifest amendment;
- `25M-03`: chain-only mode can replace parameter rows while retaining a full
  dataset's old data vectors — `data-generation-and-cuts.md`, units 8/82;
- `25M-04`: the direct tune driver silently forks the historic Optuna study
  name after the family-default repair — `training-stack.md`, unit 53; and
- `25M-05`: sweep tables record the raw optional activation flag instead of
  the resolved activation that trained — `training-stack.md`, resolved-record
  campaign; and
- `25M-06`: `.ranges` serialization collapses float32-distinct legal bounds —
  `data-generation-and-cuts.md`, canonical-representation campaign;
- `25M-07`: **RETRACTED** after the ambiguous `h` notation was challenged;
  the proposal assumed signed finite-difference arms were forbidden without
  proving CAMB's contract. Here the code's local `h` meant a numerical step,
  not cosmological `h=H0/100`; `families-scalar-cmb.md` retains the full
  correction and the number is retired;
- `25M-08`: tiny positive CMB stencil steps round to no perturbation and
  certify zero non-Gaussian covariance — `families-scalar-cmb.md`, same unit;
- `25M-09`: deleting the optional `pce`/`transfer_base` group changes artifact
  composition while strict model loading passes —
  `artifacts-inference-warmstart.md`; and
- `25M-10`: raw staging/geometry/worker prints violate every driver's
  `--quiet` all-stdout contract — `training-stack.md`; and
- `25M-11`: independently nonnegative TT/TE/EE noise amplitudes can produce an
  indefinite joint covariance — `families-scalar-cmb.md`, covariance-input
  unit; and
- `25M-12`: finite beam/noise inputs can overflow the derived covariance and
  still publish — `families-scalar-cmb.md`, same unit; and
- `25M-13`: BAOSN unions incompatible Hubble/D_M input domains and serves one
  stitched background from two cosmologies — `families-background-mps.md`,
  unit-75/fixed-facts campaign; and
- `25M-14`: a width-one transformer token makes the correction
  input-independent while the trunk can hide the demotion —
  `models-and-designs.md`, model-value-schema unit; and
- `25M-15`: streaming memory planning charges a packed target as if it had
  model-output width and can select an unsafe chunk — `training-stack.md`,
  memory-accounting campaign; and
- `25M-16`: runtime-loaded or source-inspected Python is absent from populated
  gate manifests, so adapter/driver edits retain a current PASS —
  `gates-and-board.md`, manifest-completeness campaign; and
- `25M-17`: deleting `dv_geometry/const_mask` silently disables a valid MPS
  low-k pin while schema-v2 rebuild and strict weight loading succeed —
  `families-background-mps.md`, unit-63/unit-96 interlock; and
- `25M-18`: the manifest waiver validator accepts one child file as coverage
  for an arbitrary dynamic-import tree, and board-selftest blesses the
  backwards ancestor relation — `gates-and-board.md`, manifest-validation
  campaign; and
- `25M-19`: `evaluate_yaml` is executed repository-relative but hashed
  process-CWD-first, so an outside-repo launch authenticates no input while
  executing the real file — `gates-and-board.md`, input-manifest campaign;
  and
- `25M-20`: resume skips a current downstream PASS before checking that its
  prerequisite is current, so a stale dependency can yield return code zero
  with no gate bodies executed — `gates-and-board.md`, immediate unit-4
  board-truth reopen; and
- `25M-21`: prose-only `_help` edits in board_config change every populated
  gate input digest and force false stale-input GPU reruns —
  `gates-and-board.md`, unit-4 input-identity campaign; and
- `25M-22`: `saved_emulator_root` is documented as selecting an artifact but
  no code reads it; changing it affects only the digest — `gates-and-board.md`,
  config-surface truth campaign; and
- `25M-23`: the board-listed finite-contract check calls `_chi2_domain` in
  Part H without importing or defining it, so the Torch check crashes late
  with `NameError` before Part J and its final verdict — `gates-and-board.md`,
  gate-fixture binding repair.
- `25M-24`: `run_board.main` returns from `--list` / `--check` before selector
  validation, so list+unknown gate/from/force commands exit 0 and list+check
  silently chooses one action — `gates-and-board.md`, action/selector truth.
- `25M-25`: `select_gates(--from triangle-shading)` filters out the explicitly
  named optional starting gate and begins at the next non-optional gate —
  `gates-and-board.md`, optional-start selection truth.
- `25M-26`: a child's PASS stores no dependency-result identity, so forcing a
  prerequisite in one board process and selecting the child in a later process
  resumes the old child PASS without testing the newly produced artifact;
  `--list` and `BOARD.md` publish the same false green — `gates-and-board.md`,
  immediate 25M-20 persisted-lineage reopen.

No number in this series may be reused. Architect adjudication may fold a
finding into an existing unit, but the registry retains the Red Team label and
original evidence.

### D6 triangle implementation record and temporary-directory ownership boundary

Branch `codex/d6-triangle-cleanup` started from main `b3e91b8`. The Red Team
claimed only `gates/checks/gt_b_triangle.py`, as permitted by the Wave-3
transfer. The child now identifies every triangle Axes by its x and y
parameter, traces the real z-order-zero `contourf` calls, identifies each mask
from an independent physical-formula table, and compares the exact
`(x parameter, y parameter, window)` set. It checks every collection and patch
against `_CUT_GREY`. It also locates the `omegamh2` diagonal specifically and
checks both excluded interval endpoints. The mutation moves the sole
`omegabh2` artist from `(h0, omegab)` to the unshaded `(ns, h0)` panel. The old
global summary remains exactly `(12 artists, 7 shaded panels, 12 gray
artists)` before and after the move, while the exact-set predicate reports
three errors and rejects the figure.

Direct Mac/CPU evidence under the Cocoa interpreter: all four child legs pass,
the mutation arm reports equal old counts and a rejected exact set,
`py_compile` passes, and `git diff --check` is clean. A 130-dpi rendering was
inspected: the six expected two-dimensional panels and the `omegamh2`
diagonal carry the gray cut regions, while unrelated panels remain unshaded.
The bounded C-reader follow-up replaced every new list/set/generator
comprehension and every generator-based `extend` or `all` call with named
ordinary loops. Multi-argument helper and acceptance calls put one argument
on each line where practical. An AST scan has zero comprehension or generator
nodes and zero generator-based `extend`, `all`, or `any` calls. The four gate
results plus the mutation counts are unchanged.
Queue-2 fan-out batch 1 later landed the original four aid bindings in
`b9244cf`. This branch then merged that main-line change, retained its
`report(aid=...)` mechanism and changed the four declarations, emissions and
home-note anchors together to the exact-owner claims above. The direct child
still passes all four legs after that merge. A board-wrapper attempt stopped
in preflight because the Mac lacks CosmoLike, CUDA and a resolved ROOTDIR. No
gate body ran, so a board-level verdict is not claimed.

The D6 temporary-directory half did not enter this landing because its files
have other active owners. The original nine-site contract is exact at the
adjudication commit `cbd0a9c`: five sites are
`finetune_identity.py` (`ftw-`), `gct_parity.py` (`gct-`),
`gsv_bitwise_drift.py` (`gsv-`), and both `transfer_identity.py` sites
(`tpe-e-`, `tpe-`). The other four are the then-present
`board_selftest.py` sites with prefixes `board-selftest-`, `board-yaml-`, and
`board-logtrust-`, plus the empty manifest-directory fixture created under
`gates/`. Those files, along with `board.py` and `run_board.py`, were expressly
outside this claim. Their owner must complete the nine-site context-manager or
failure-safe cleanup landing and its injected-failure proof. Later additions
to `board_selftest.py` require a fresh census by that same owner; they do not
silently change which nine sites this contract names.

Architect adjudication overlay for the first batch (received after commit
`fafc122`): all six were confirmed. `25M-01` mints unit 94 and couples
generation-side support truth to unit 84's inference side; `25M-02` and
`25M-03` amend unit 8, with `--chain` now an explicit mode axis; `25M-04`
amends unit 53; `25M-05` extends unit 41 to sweep products; and `25M-06`
extends unit 82 with unit 87's decimal-contract coupling. The Architect also
adopted two process lessons: forward-walk the real signature after a repair,
and file identity cannot substitute for representation truth.

Architect adjudication overlay for `25M-14`/`25M-15` (received after commit
`efddf98`): both were confirmed without amendment and both workstation gates
were commissioned on the filed terms. `25M-14`'s width-one transformer
input-independence was accepted as the strongest silent-demotion proof;
`25M-15`'s Torch integration and pure 84-byte/refusal arithmetic join the
queue-5 workstation exhibits. DIDACTICS-93 is ratified as binding law, with
the Planck-LCDM covariance byte-identity control. The series at that
adjudication stood at 01--15 with retired tombstone 07.

Architect adjudication overlay for the manifest/artifact batch: the first
reply explicitly confirmed `25M-16` through `25M-21`; a separate reply then
confirmed `25M-22`. Queue 2 is resequenced behind one hardening increment.
The Architect recorded one process correction against the population audit:
it verified declarations through the same scanner that required independent
attack, so “audit the validator, not only through it” is binding law. The
presence-inference census closure was accepted with `const_mask` as the only
remaining silent site. For `25M-22`, the recommended removal branch is
binding: remove `saved_emulator_root` plus its help, and make the census that
every public non-documentation board-config key has an execution reader a
permanent selftest leg. The hardening landing is therefore 16 whole-check
closure + 18 all-quantified waiver coverage + 19 owner resolvers + 21 digest
projection + 22 dead-key removal, with one deliberate digest transition.

Architect adjudication overlay for `25M-24`/`25M-25` (received after commit
`97e8802`): both were confirmed without amendment and join the 1b hardening
increment as items 7 and 8.  The Architect adopted the promise-bypassed-by-
ordering classification for 24 and requested the remaining `main()`
early-return sweep.  That sweep found no other non-duplicate early-return
false green in the check scripts or public drivers.  One 25M-24 acceptance
rider remains for adjudication: a valid named `--force-rerun` outside the
resolved selector surface is silently discarded and must be rejected as an
ignored control, not counted as a new finding.

25M-26 corrects an overstatement in this register's RT-IMPL-01 audit.  The
landed `_resume_state` unifies a gate's **own** code/input/log currency, but it
does not unify persisted dependency-result lineage: the accepted 25M-20 fix's
`reran` set lives for one invocation only.  The durable two-command witness and
repair contract are in `gates-and-board.md`; this correction preserves the
valid raw-log acceptance while withdrawing the broader claim.

Architect adjudication overlay for `25M-26`: CONFIRMED without amendment and
assigned as hardening item 9.  The two-invocation witness is permanent;
`stale-dependency` is a first-class resume state; legacy dependent records
without a snapshot are non-green.  The adjudication also creates a standing
audit rule: every pre-merge review reconciles the issued ruling's complete
clause list against the implementation diff and its discriminating gates.
Reviewing only the coherent subset the implementation chose to deliver is not
acceptance.  The hardening increment is closed to unrelated expansion; a new
item joins only by reopening an already-issued clause.

Clause-reconciliation batch after the 25M-26 audit law (all Red Team
CONFIRMED, awaiting Architect adjudication):

- `25M-27`: queue 1c builds its root-driver Git pathspec from currently
  existing `*.py` files, so deleting a tracked driver makes it disappear from
  the dirty-tree question and preflight reports clean — `gates-and-board.md`.
- `25M-28`: 1b promised stale-member inspection in both `--list` and
  `BOARD.md`; list omits it, and byte-identical input relocation loses the
  name because `_stale_member` compares only hashes, not paths —
  `gates-and-board.md`.
- `25M-29`: unit 14(f)'s mandatory extreme-scale eval Part I was never
  committed; the board cannot catch restoring the float32 mean overflow —
  `training-stack.md`.
- `25M-30`: unit 14(h) live-drives only CMB; grid and grid2d corrupt-score
  refusals were replaced by an AST call-name census — `training-stack.md`.
- `25M-31`: the finite-contract gate's Parts A/C assert retired error prefixes
  and remain false-red even after the separate 25M-23 import repair —
  `training-stack.md`.
- `25M-32`: `_grid2d_law_rows` overwrites queue 3's canonical seeded loader
  order with sorted compact `arange`; the MPS gate asserts the defect as truth
  — `data-generation-and-cuts.md`.
- `25M-33`: bounded staging checks only that a base covers the selected
  maximum row, not exact raw/base row-count equality; shifted same-width files
  produce finite wrong log-law targets — `data-generation-and-cuts.md`.
- `25M-34`: stored-float32 moments are implemented correctly but the gate has
  no payload where the forbidden pre-cast accumulator changes the pin mask —
  `data-generation-and-cuts.md`.
- `25M-35`: the failed-sweep lifecycle leg manually calls release and never
  executes `_sweep_job`; deleting real driver cleanup stays green —
  `data-generation-and-cuts.md`.

This batch is the first direct application of the clause-checklist law.  Each
item cites an issued clause and the exact missing implementation or
discriminating leg.  Current-correct producers are stated as such for 25M-34
and 25M-35; their defect is the claimed acceptance evidence, not the runtime
body.  No number in the series may be reused.

Architect adjudication overlay for `25M-23` and `25M-27` through `25M-35`
(received after commit `81183e7`): all ten findings were CONFIRMED.  The
Architect's audit recovered `25M-23` from this durable register even though no
chat relay had carried it; that is direct acceptance evidence for the rule
that the file record, not chat, is authoritative.  The clause-reconciliation
classification is adopted: issued clauses with missing or substituted
acceptance evidence are defects even when the current producer is correct.
In particular, the precise wording for `25M-34` and `25M-35`—correct
producers, defective claimed evidence—is binding.

`25M-32` is the batch's heaviest item and reopens queue 3.  Its repair and the
gate assertion that currently blesses sorted `arange` order land atomically;
the gate must be flipped in the same unit so it cannot continue to encode the
defect as truth.  The machinery follow-up `25M-27`/`25M-28` queues behind the
in-flight census-core remainder.  Queue 2 remains closed until the remainder
and that follow-up are green.  At this adjudication the series stands at
`25M-01` through `25M-35`, with the single retired tombstone `25M-07` and no
unreconciled chat-only finding.

## Red Team implementation record: D2 factual increment

This increment implements the collision-safe factual and teaching corrections
transferred to the Red Team under the USER REASSIGNMENT.  Architect acceptance
is still required; this record is implementation evidence, not self-
certification.

The public-documentation changes are:

- the root README uses the user-defined SONIC expansion, describes the actual
  bounded staging and artifact/device behavior, and distinguishes the common
  per-row score interface from each family's different scientific metric;
- the dense-layer and residual-block chapter now distinguishes dimension-
  changing projections from fixed-width residual blocks and shows the executed
  order: second linear, skip addition, final normalization, final activation;
- warm-start prose distinguishes fine-tuning's numerical parity check from
  transfer's bitwise check and explains the optional anchor as a decoupled
  pull toward saved weights, not as part of the scientific loss;
- the PCE text defines the finite Legendre-polynomial base and its neural
  residual refiner without calling the base a neural trunk;
- the CMB chapter has separate spectrum-generation and covariance-generation
  branches, spectrum-specific covariance equations, the actual amplitude-law
  transform, the twice-applied roughness operator, and the narrow units served
  by the current adapter.  Run history and internal ledger language are
  absent;
- the matter-power chapter distinguishes the three target laws and states that
  Syren base sidecars are conditional on `write_syren_base`;
- `emulator/README.md` now teaches the five family-specific score meanings,
  current dependencies, producer/consumer ownership, conditional Syren base
  files, CPU-normalized artifacts, and current architecture scope; and
- `syren/README.md` defines its inputs, outputs, units, three target laws, and
  present vendoring relationship without claiming byte identity or recounting
  development history.

The in-file documentation changes name tensor shapes, ownership boundaries,
copy/view behavior, device and dtype transitions, state-dictionary meaning,
active-source coordinates, CMB factor directions, and PCE mechanics in the
Python voice defined by `user-didactics-and-python-voice.md`.  Executable AST
structure is unchanged.  Four files also receive punctuation or a clearer
current refusal message in user-facing exception strings:
`emulator/losses/cmb.py`, `emulator/results.py`, `emulator/training.py`, and
`emulator/warmstart.py`.

### Deliberate holds and later owners

DIDACTICS-79 is **not claimed closed** in this increment.  The old generator
command was false: it omitted required arguments, used an invalid boundary,
and named project files absent from this checkout.  The invalid command is
removed, and the invocation choices now state that `--unif` and `--seed` are
required and that `--boundary`, when supplied, is interior.  No replacement
command is printed because no shipped minimal generator configuration can be
executed in the available environment.  The command, minimal per-family YAML,
and rank/worker/checkpoint teaching remain open under DIDACTICS-79; the
executed-before-printed rule forbids inventing an example.

D1 navigation prose, the cohesive current-state visits that remove remaining
README diary language, gates/check teaching after D4+D5, and lane-3 protocol
prose remain with their named later increments.  This bounded factual landing
does not claim those surfaces.

### Evidence collected before Architect audit

- `python3 -m compileall -q emulator cobaya_theory syren
  compute_data_vectors gates/checks` completed successfully;
- `PYTHONPATH=. python3 gates/checks/board_selftest.py` completed with its
  all-pass verdict;
- a Python AST comparison found no non-string executable change in any edited
  Python file;
- `git diff --check` reported no whitespace errors;
- Markdown fence, math-delimiter, table-column, relative-file-link, and local-
  anchor scans found no unresolved item after the dense-layer anchor repair;
- an untruncated scan found no retired SONIC expansion; and
- the generator parser was executed in isolation: valid explicit `--unif 1`
  and `--seed 17` values were retained, while omission of either required
  option returned the parser's error status.  This parser proof does not stand
  in for the still-open end-to-end DIDACTICS-79 command.

The remaining history-vocabulary candidates in the four READMEs are
pre-existing and belong to the later current-state file visits.  They are not
silently blessed by this increment.

## Red Team implementation record: root README code-map deduplication

The root README linked to `emulator/README.md` twice in immediate succession:
first in a complete sentence explaining what the package code map contains,
then again as the first item under `Contents`.  The second link added no new
navigation or teaching value.  This bounded follow-up keeps the explanatory
sentence and removes only the duplicate contents entry.  The numbered table
of contents and the code-map destination are unchanged.

Evidence before Architect audit: an untruncated scan finds one `code map`
navigation reference in the root README, the retained relative link resolves
to `emulator/README.md`, and `git diff --check` is clean.  This is
implementation evidence, not Red Team self-certification.

The same follow-up removes development-ledger language from the saved-artifact
warning.  The README now states the action a user must take: keep the `.h5`
recipe and `.emul` weights together, never combine members from different path
roots, and never replace only one member of a trusted pair.  It no longer
describes the queued digest or publication implementation.  That engineering
contract remains in its owner note rather than in the public introduction.

## Red Team implementation record: root README current-state and figure reuse

This increment applies the user's ruling that the public README presents the
library as it exists.  It is not a diary of rejected designs, gate runs, queue
positions, or future repairs.  It also applies the user's prose ruling that a
full explanatory clause belongs in a sentence or a table, not inside
parentheses or between em dashes.  Literal function calls, tensor shapes,
mathematical grouping, coordinate pairs, and compact code labels remain
parenthesized where the punctuation is part of the notation.

The root README now:

- introduces each path flag, data file, training control, model component,
  warm-start mode, scientific family, and precedence rule in direct
  current-state prose;
- separates choices that had been compressed into parentheses into named
  sentences or table columns, including the five data files, the `name` and
  `ia` model axes, fine-tuning versus transfer, generator sampling modes, and
  phase precedence;
- retains warnings when they change what a user should do now, while removing
  internal note paths, queue language, run-history evidence, and discussions
  of implementations that are not part of the public interface;
- defines the fixed-width residual block separately from the entry and exit
  projections that change dimension;
- replaces the activation ASCII sketch with the manuscript's three-panel
  activation figure and preserves the forward-versus-backward warning at
  exact zero;
- reuses the manuscript ownership-chain figure and defines every table and
  dimension symbol introduced by its first box, including NumPy's disk-backed
  `memmap` mechanic;
- reuses the two-parameter coverage figure, defines its axes as arbitrary
  sampled cosmological parameters, and states explicitly that the validation
  region is a rule described in prose rather than a third region drawn in the
  figure; and
- links every PNG preview to its vector PDF.  The vector sources remain owned
  by `texnotes/make_figures.py`, while the new
  `texnotes/render_readme_previews.py` deterministically renders the three
  browser previews at 180 dpi without changing their aspect ratios.

This increment touches public documentation, three derived image assets, and
the preview-rendering script only.  It does not edit `emulator/` production
code, the board, or gate checks.

### Evidence collected before Architect audit

- `python3 texnotes/render_readme_previews.py` regenerated all three requested
  previews from their vector PDFs;
- an immediate second render reproduced the same SHA-256 digest for every PNG;
- `python3 -m py_compile texnotes/render_readme_previews.py` completed without
  an error;
- the regenerated dimensions are 1800 by 550 pixels for the ownership chain,
  1800 by 525 for the coverage figure, and 1800 by 700 for the activation
  figure, exactly preserving the three PDF aspect ratios;
- visual inspection at original resolution found no cropped labels,
  overlapping legends, or stretched axes;
- a local-anchor scan reconciled all 62 unique fragment targets with their
  headings or explicit stable anchors;
- a relative-link scan outside code fences found no missing file, figure,
  script, or anchor destination;
- the 166 fenced-code delimiters form 83 complete pairs;
- an untruncated scan finds no em-dash character and no remaining
  clause-bearing parenthesis candidate outside the deliberately retained PCE
  degree tuple; and
- `git diff --check` reports no whitespace error.

These checks are implementation evidence for the Architect's pre-merge audit.
They are not Red Team self-certification.
## Red Team audit: manuscript public-prose and typesetting pass

This is an audit record only.  It changes no TeX source, generated figure, or
PDF.  The manuscript remains readable and detailed, but it is not ready to
close under the public-prose constitution.  The following findings require an
independent Architect ruling before implementation.

### TEX-PROSE-01: one malformed unit expression survives compilation

At `texnotes/emulator_code_guide.tex:4969-4974`, the source uses `\ {` before
literal `m km...` and `m Mpc` text.  TeX accepts the source but renders stray
`m` characters.  The quantities need explicit roman unit expressions such as
`\,\mathrm{km\,s^{-1}\,Mpc^{-1}}` and `\,\mathrm{Mpc}`.  Acceptance requires a
clean compile plus visual inspection of the affected PDF page.

### TEX-PROSE-02: prose dashes and corrective-negation frames remain

Prose dash candidates remain at source lines 206, 364, 530-533, 848, 1919,
1995, 2085, 3160, 3444, 4061, 4213-4214, 4602, 4622-4624, 4759, 4912, 4919,
and 6613-6614.  Command-line options beginning with `--` are syntax and are
excluded.  Corrective-negation clusters remain at 62-68, 889-900, 1313-1317,
1580-1584, 2027-2064, 2161-2166, 2619-2622, 2683-2685, 2994-3008,
3253-3264, 3839-3844, 4197-4200, 4338-4358, 4554 onward, 4765-4771, 4993
onward, 5083 onward, 6283-6295, 7381-7388, and 7598-7603.  Mathematical
negation and direct refusal rules are outside this finding.  The repair states
the positive scientific or operational claim directly.

### TEX-PROSE-03: development-state narration interrupts the guide

Approximately 58 paragraphs titled or introduced as `Current gap`, `Required
closure`, or `Current deviation` remain between the first instance near
919-927 and the final cluster near 6228-6237.  The manuscript should teach the
current behavior, the consequence, and the safe action.  Queue history,
future-repair specifications, rollout narration, and landing evidence belong
in `notes/`.  The policy discussion at 738-751 and landing narration at
5947-5950 are additional instances.

### TEX-PROSE-04: the gate appendix repeats 120 inline pseudo-headings

Lines 6411-7069 repeat `Claim and path`, `Fixture and verdict`, and `Catch
power` for 40 gates.  The three fields are useful and must stay, but a single
defined gate-description environment or compact table should own their
typography.  Acceptance preserves every factual field while removing the
repeated inline-heading cadence.

### TEX-PROSE-05: several mathematical symbols enter before a stable definition

- Lines 2814-2817 use PCE half-width `h_j`; the symbol should be renamed because
  `h` already denotes the cosmological quantity `H_0/100`.
- Lines 2862-2869 and 3004-3008 use `q` for both the PCE sparsity exponent and
  the retained SVD rank; the rank needs a distinct symbol.
- Line 4311 needs to define `e` as the epoch number.
- Lines 4396-4405 need to define `y_0`, the residual `r`, and the composed
  prediction `y` before using their composition equation.
- Lines 4626-4642 need the physical meanings of `a_1`, `a_2`, and `b_{TA}`
  before listing the TATT monomials.

### TEX-PROSE-06: external algorithm claims lack an authoritative citation

The source contains no `\cite` command.  Cobaya, CAMB, CosmoLike, ensemble
moves, Legendre PCE, LARS/OMP, PRESS leverage, BerHu, AdamW, NLA/TATT, and
Chan/Welford statistics are the priority citation surfaces.  A claim that is
specific to this repository may instead be narrowed explicitly to `as
implemented here`.

### TEX-PROSE-07: passive prose hides the owning program boundary

Representative sites are 913-916, 1367-1369, 1904, 3976, 4175-4177,
4231-4234, 4926, 5429-5434, and 5633-5637.  Each repair names the class,
function, adapter, or driver that validates or changes the state.

### TEX-PROSE-08: broad gate verbs exceed the executed fixture

The strongest clusters occur at 5755-5758, 5865-5874, 7592-7598,
7642-7694, and 7778-7787.  Verbs such as `prove`, `ensure`, `close`, and
`prevent` need to become the exact fixture and observed verdict, followed by
the boundary the gate does not establish.

### Recommended repair order

Repair TEX-PROSE-01 and TEX-PROSE-02 first because they are mechanical hard
failures.  Remove the diary layer next.  Then refactor the gate appendix and
revise definitions, citations, named owners, and gate-claim precision.  The
file-study itinerary should remain detailed; its list structure helps a new
reader.  A complete compile, PDF render, visual inspection, link scan, and
repeat public-prose census are required after the edits.

## TEX-PROSE-01 and TEX-PROSE-02 implementation evidence

The landing edits `texnotes/emulator_code_guide.tex`.  TEX-PROSE-03 diary
removal and the TEX-PROSE-04 gate-appendix refactor remain separate work.
Mechanical TEX-PROSE-02 edits apply throughout the source, including prose
inside those later sections, but their structure and evidence fields remain
unchanged.

The malformed background-function units now render as
`70 km s^-1 Mpc^-1`, `13296.826 Mpc` and `15538.408 Mpc`.  The source uses
explicit roman unit expressions.  Rendered page 50 has no stray `m` prefix.

The repeated census produced these counts for guide prose:

- unescaped semicolon characters: 0
- Unicode em dash, en dash, ellipsis and curly quote characters: 0
- TeX prose `---` sequences: 0
- question marks: 0
- multiline `,\s+(and|or)` candidates: 9 after adjudication; the 21 serial
  lists were repaired and the retained candidates join independent clauses or
  two objects separated by an appositive
- editorial pass against private standards: zero matches
- bold-first instructional bullets: 0

Remaining double-hyphen sequences belong to literal command-line options.
Escaped backslash-plus-semicolon sequences are TeX math-spacing commands,
which remain outside the punctuation count.
The repeated bold evidence labels in the gate appendix remain queued under
TEX-PROSE-04.

Preservation checks compared the edited guide with its pre-edit version.  The
ordered equation-environment sequence is identical.  Label, equation-reference
and included-figure counts are identical.  The numeric-token multiset is
identical after normalizing the old TeX range spelling.  Equation bodies have
three intended text-level changes: the two punctuation removals and one
`bin-angle` spelling repair.  Brace balance is zero and `git diff --check`
passes.

Two `pdflatex` passes completed in `/tmp/tex-prose-0102`.  The final PDF has
84 pages.  Its log has no LaTeX warning, unresolved reference, overfull box,
multiply defined label or undefined control sequence.  Existing underfull-box
diagnostics remain.  Raster inspection covered the title page, activation
figure and comparison table, the repaired unit page and representative gate
appendix pages.  The final sample comprised pages 1, 3, 6, 12, 18, 24, 28,
35, 41, 44, 50, 55, 62, 68, 73, 80 and 84.  The inspected pages have complete
text, intact equations and no clipping or overlap introduced by this landing.

The tracked `texnotes/emulator_code_guide.pdf` was rebuilt from the edited
source.  A first invocation from inside `texnotes/` failed before page 1
because the manuscript names its artwork relative to the repository root.
The successful two-pass invocation therefore ran from the repository root,
which is the required build context.  The failed attempt is recorded here so
the evidence does not imply that both working directories are supported.  A
second mechanical source census after that build found no remaining listed
token patterns.  The independent audit below later found semantic defects
that this mechanical scan could not detect.  The scan also confirmed that the
ordered labels, references, environments, equation environments and
numeric-token multiset still equal the pre-edit source.

### Independent-audit hold and repair

The first candidate was held after an independent source and rendered-page
audit.  Five prose edits had changed scientific or Python meaning.  The held
text said that reshape preserves axes, described the chi-square form rather
than the rejected width-squared tolerance, turned a weight-decay exemption
into frozen parameters, separated focus weights from the errors that define
them and grouped epoch loss with gradients and parameter deltas.  The repair
now states the exact invariant or owner in each case.

The same audit found duplicated phrases in the Stage-1 file-study table and a
stray word in the EMA description.  Both visible defects are repaired in the
source.  The initial line-local comma scan had also missed conjunctions on the
next line.  A whole-file multiline scan found 30 candidates.  Human
adjudication identified 21 serial lists and nine grammatical false positives.
The 21 final serial commas are removed.  The nine retained candidates connect
independent clauses or separate two objects around an appositive, so they are
outside the Oxford-comma rule.  The repeated scan uses the exact pattern
`,\s+(and|or)\b` over the complete TeX source.

This repair record preserves the failed first audit rather than replacing it.
The branch remains implementation evidence for Architect review and does not
certify the manuscript.

After the repair, two `pdflatex` passes from the repository root wrote
`tmp/pdfs/tex-prose-repair/emulator_code_guide.pdf`.  The output has 84 pages
and 3,929,660 bytes.  The final log has zero LaTeX warnings, undefined
references, multiply defined labels, overfull boxes, undefined control
sequences or fatal errors.  The 291 existing underfull-box diagnostics remain.
Rendered pages 6, 35, 38, 39, 40, 50, 66, 73, 81 and 84 were inspected after
the semantic and table repairs.  Text is complete and the repaired Stage-1
table has no repeated phrases or overlap.  The tracked PDF and the fresh build
share SHA-256
`3c32568093dfba956565163a10ebc01df497e60122a548bc08071c80968d9f17`.

## Red Team implementation record: second root-README public-prose pass

The Architect cleared the preceding figure and didactic increment at `701d6f9`.
This follow-up applies the public-prose constitution to the complete root
README, so it requires a new Architect audit before the branch is merge-ready.

The revision keeps the README's reference structure, equations, warnings,
examples, and undergraduate-level explanations while changing the sentence
forms that interrupted them.  In particular, it:

- removes prose em and en dashes, curly quotation marks, canned conversational
  phrases, and corrective-negation formulas;
- replaces contrast-first explanations with direct definitions of data
  staging, residual blocks, normalization features, attention, PCE modes,
  warm starts, CMB conventions, background distances, grid2d target laws,
  generator checkpoints, and the pipeline's storage coordinates;
- turns the activation generalizations and saved-source choice into real
  subsection headings, while retaining the equations and examples beneath
  them;
- states current warnings as the condition and safe user action, without
  narrating rejected implementations or repair history;
- distinguishes the network's hidden features from parameter, data-vector,
  and batch axes in positive terms;
- states that the saved artifact owns rebuild configuration and that the
  sampling YAML supplies only the artifact root and runtime device;
- gives the direct-programming appendix a schema-v2 description instead of a
  historical comparison; and
- replaces the vague authorship sentence with the concrete roles of Claude
  Code and Prof. Miranda.

### Evidence collected before Architect audit

- the README preview renderer reproduced all three browser images, and Git
  reports no image-content change;
- `python3 -m py_compile texnotes/render_readme_previews.py` completed without
  error;
- the README contains 83 balanced fenced-code pairs;
- all 11 relative file targets exist;
- all 62 unique local fragment targets reconcile with explicit anchors or
  established heading slugs;
- all 44 Markdown tables retain a consistent column count;
- untruncated scans find no prose em or en dash, curly quotation mark, listed
  canned conversational phrase, or contrast formula using `rather than`,
  `instead of`, `unlike`, `by contrast`, `not just`, or `not only`;
- the two exact generator-move weights in the README were checked against
  `compute_data_vectors/generator_core.py:713-714`; and
- `git diff --check` reports no whitespace error.

The remaining negative clauses are direct safety or exclusivity statements:
the alpha warning, artifact-pair protection, and two mutually exclusive target
constructions.  None introduces a corrective positive claim.  These checks
are implementation evidence for the Architect's review, not Red Team
self-certification.

### Stricter public-prose follow-up

The user expanded the public-prose rules after the preceding record was
written.  This follow-up applies those rules to the complete root README.  It
removes the remaining semicolons, Oxford commas, question-shaped diagram
labels and typographic quotation or dash characters.  Independent clauses
were split or joined with a precise relation.  The Oxford-comma pass was not a
blind punctuation deletion.  Scientific lists such as TT, TE, EE and pp keep
every required member.

The same pass checks four broader prose classes.  It removes one banned
importance word, `robustness`, and replaces it with the specific
`outlier-control` description.  It rewrites trailing `allowing` and
`Providing` clauses with named subjects.  It removes inline `Why:` and
`Spellings:` labels in favor of ordinary sentences.  It also removes bold
from warning-like declarations and numbered definitions when the typography
did not identify a new technical term.  Bold remains for the SONIC acronym,
the top alpha warning and first definitions of terms such as emulator,
residual block and saturation.

The Optuna journal sentence remains tied to the code in this branch: an
existing journal resumes the named study.  Unit 53's study-manifest behavior
has not landed here.  Its eventual code landing must update that README
sentence in the same change so the public current-state description never
leads or trails the implementation.

The final untruncated census records a zero-match editorial pass against
private standards.  It also reports zero matches for:

- em dash, en dash, curly quotation mark, Unicode ellipsis, semicolon,
  question mark and comma followed by `and` or `or`; and
- bold-first list items and inline-header list items.

Markdown checks find 166 fence delimiters, which form 83 pairs, and 44 tables
with a consistent column count inside each table.  All nine unique relative
file destinations exist.  The 62 unique local fragment destinations still
resolve through explicit anchors or the established GitHub heading forms.
`git diff --check` reports no whitespace error.  These results are
implementation evidence for Architect review and do not certify the landing.

### Independent factual review of the rewritten README

A second agent compared every substantive README change with the current
implementation.  That review found five scientific or code-contract
regressions plus five ambiguous sentences.  All ten were corrected before
the landing was presented for audit.

The corrected scientific statements now say:

- CMB roughness adds `lambda * c_rough` to the training score before the
  configured loss transform, while evaluation reports the plain chi-square;
- Syren artifacts store a law name without a formula digest, so an edit to a
  vendored formula requires retraining existing artifacts;
- crossing activation learning curves give a training-size-dependent ranking,
  while coincident curves are a tie;
- `p_max`, `r_max`, and `q` define the PCE candidate basis, `max_terms` limits
  the tested active support, and `loo_max` accepts or rejects a mode; and
- `trim` drops the worst-fit fraction while `focus` upweights difficult rows
  that remain in the sample.

The clarity repairs distinguish trainable LayerNorm affine parameters from
the transformer's fixed choice of normalization type, define attention weights
as functions of the current sample's tokens, restore a complete sentence in
the shared-MLP paragraph, repair the basis-change diagram, and remove a
tautological Optuna evidence item.

The factual reviewer also found a separate deferred-import documentation
follow-up in `emulator/README.md`: several lines still imply that importing
`geometries/output.py` loads CosmoLike.  That file is outside this root-README
landing.  Its package-README visit must state the landed `25M-37` boundary:
the optional imports now occur inside `from_cosmolike` and
`build_shear_angle_map`, while importing the module and using other geometry
paths remain clean.

Preservation evidence from the independent review: every display-math block
is byte-identical to the pre-rewrite source, all fenced blocks retain their
order and language, changed fence content is explanatory text rather than an
executable command or YAML value, and `git diff --check` is clean.  The review
is evidence for the Architect and does not certify the landing.
## Queue-2 note-side evidence draft and new correctness finding

The Red Team drafted the A1-ii home-note surface for the 27 gates outside the
Implementer's seven foundation gates and four wrapper surfaces (six board
gates).  The six existing home notes now contain one six-field block per gate
plus one narrowed, long-`Gate.id` anchor per logical leg.  The draft has 137
leg anchors.  Exact
dot-to-dash mapping, anchor uniqueness, six-field completeness, and declared-
name reconciliation were checked mechanically.  Logged-only, visual,
conditionally absent, environment-owed, and currently red claims are labeled
instead of being promoted to green evidence.  These blocks are a draft for
Architect audit; they do not certify the future runner wiring.

The execution pass also found `25M-36`, recorded in full in
`families-background-mps.md`: `mps-identity` is currently false-red because
its bounded-staging mean leg compares the streamed float32-payload mean with a
mean formed before the independent law rows are cast to float32.  The producer
matches the correctly ordered independent stored-payload reference exactly;
the gate does not.  The producer remains frozen, the wrong-reference mutation
must become a discriminating red leg, and the current queue-2 block explicitly
withholds a whole-gate pass.

The same execution pass found `25M-37`, recorded in
`artifacts-inference-warmstart.md`: four gates declared as Torch-only fail
before their first assertion because importing the persisted output-geometry
type eagerly imports the compiled CosmoLike interface.  The fix belongs at
the production `from_cosmolike` boundary so constructor/from-state artifact
use is genuinely independent; adding four gate-local stubs would preserve the
false public import contract.

The DIDACTICS-79 execution pass found `25M-38`, recorded in full in
`data-generation-and-cuts.md`.  A real two-rank, one-parameter background run
loads Cobaya and CAMB, then rank zero writes `# weights lnp H0` as the first
line of its `.ranges` file.  GetDist 1.7.2 treats that four-token comment as a
range record and raises while converting `weights` to a float.  The run
returns status 1 before covariance or data-vector publication.  All four
generator drivers inherit the writer.  The exact affected set is fresh
one-parameter runs in either sampling mode and either chain mode.  Wider
headers have five or more
tokens and this GetDist version ignores them.  DIDACTICS-79 remains held until
the new correctness finding is adjudicated and the real command is replayed.

## Red Team implementation record: 25M-37 evidence readback and Torch probe

The audited production repair at `3ba8588` defers the optional geometry
dependencies to the two operations that use them.  `from_cosmolike` owns the
compiled CosmoLike interface plus GetDist's `IniFile`, while
`build_shear_angle_map` owns `IniFile` alone.  Importing
`emulator.geometries.output` no longer loads either optional package.

The two existing queue-2 evidence blocks affected by that repair now describe
the landed code and direct execution:

- `scalar-identity` reaches every declared assertion with the Cocoa Torch
  2.6.0 CPU interpreter on a machine without the compiled CosmoLike interface
  and ends `PASS: scalar-identity all checks green`;
- `finite-contract` reaches its body, records the four known Parts A/C
  message-prefix false reds, completes Parts B/D/E, then crashes in Part F
  because the synthetic loss object has no `geom` from which
  `_chi2_n_terms` can obtain a contraction width.

The queue-2 draft deliberately excluded `finetune-identity` and
`transfer-identity` with the other wrapper-family gates, so there are no
six-field blocks for those two gates in the red-team-owned note surface yet.
Their direct children now pass the repaired import boundary.  The fine-tune
child ends `finetune-identity: ALL PASS`.  The transfer child executes all 59
logical checks and retains its separate known red on the cross-family fixture,
ending with one failure.  The Implementer's wrapper-family evidence rollout
must record those current results when it creates the two excluded blocks.
This bounded update does not take ownership of their leg names.

The environment probe requested for the increment-2 seam also succeeds.  The
interpreter at
`/Users/vivianmiranda/data/COCOA/june2026/cocoa/Cocoa/.local/bin/python`
imports Torch 2.6.0.  A real `torch.nn.Linear(2, 1, bias=False)` forward pass
with weight `[3, 4]` and input `[1, 2]` prints
`torch 2.6.0 device cpu forward [[11.0]]`.  This result answers the Architect's
probe positively and makes the conditional D3 transfer executable in this
environment.  It does not claim CUDA or workstation evidence.

Evidence commands, run from the current worktree with `PYTHONPATH=.`:

```text
Cocoa/.local/bin/python gates/checks/scalar_identity.py
Cocoa/.local/bin/python gates/checks/finetune_identity.py
Cocoa/.local/bin/python gates/checks/transfer_identity.py
Cocoa/.local/bin/python gates/checks/finite_contract.py
```

The first two return zero.  Transfer returns one for its independently known
fixture red.  Finite-contract returns one at the Part F fixture crash after
the import repair lets it reach the check body.  These are implementation
readbacks for Architect audit, not self-certification.

## Red Team implementation record: 25M-36 stored-payload reference order

The Architect confirmed `25M-36` and transferred its bounded
`gates/checks/mps_identity.py` repair to the Red Team. The producer remains
unchanged. The check now creates its independent reference in the same
representation order the producer owns: calculate the law in float64, store
each row in float32, promote those stored rows for a float64 column mean and
convert the final mean to the persisted float32 dtype. Both staging fixtures
use this order, which removes a second copy of the same pre-cast reference
pattern from the smaller fixture.

The production-width fixture keeps the stored values and their mean as
separate exact assertions. It also computes the old mean-before-cast result as
a mutation control. The correct stored-payload reference is array-equal to the
producer center. The old ordering differs by `5.960464478e-08`, so replacing
the repaired reference with that ordering makes the exact mean assertion red.

A direct Cocoa Torch 2.6.0 CPU run before the edit ended with one failure,
`bounded staging: streamed mean equals the known answer`. The same complete
child after the edit reports the old-order discrimination and ends
`PASS: mps-identity all checks green`. The existing logical aid
`mps-identity.bounded-staging-values` remains the owner of the added internal
assertion. No `board.py` or runner file was touched. Queue 2 still owns the
machine wiring of that aid.

The executed command from the isolated worktree was:

```text
PYTHONPATH=. MPLBACKEND=Agg Cocoa/.local/bin/python gates/checks/mps_identity.py
```

The Cocoa interpreter compiled the edited check, `git diff --check` returned
zero and the scoped diff contains only the check, its MPS home note and this
durable register. This record presents evidence to the Architect and does not
certify the landing.

## D3 implementation blocker: disjoint fixtures expose the existing thin smoke margin

The transferred scalar-smoke visit now has a provisional implementation of
the fixture-authenticity boundary.  It stages training and validation first,
compares exact float32 physical parameter rows, independently recomputes the
scalar target from each row's own `H0` and `omegam`, and refuses any overlap
before geometry construction or training.  With generator seeds 1234 and
5678, the staged files contain 4,000 and 1,000 aligned rows with zero overlap.
The same-seed mutation contains exactly 1,000 overlapping validation rows and
is refused before training.  The implementation also contains the claimed
independent cut-banner fixture, which recovers `used 3 of 5 cut rows`, checks
the exact selected row order `[1, 3, 2]`, and rejects the hardcoded
`used 1 of 1` plus wrong-row mutation.

The seed change interacts with the existing two-epoch prediction control.
With the required train seed 1234 and validation seed 5678, all authenticity
checks pass and the validation median is finite at about 0.197, but the saved
prediction at `H0 = 73`, `omegam = 0.32` is 0.157807 versus the analytic
0.170528.  The relative error is 0.074596, above the unchanged 0.05 bar.  A
second full run reproduces the same digits because the training path uses its
own fixed seed.  For comparison only, the pre-change fixture built with seeds
1 and 2 passes at relative error 0.0465, the known thin margin.  Seeds 1 and 2
are not being adopted as an unruled substitute.

The contract's single-population alternative was tested once without changing
the training budget or bar.  One 5,000-row physical table was generated by
`numpy.random.default_rng(1234).normal` with per-row location `[70, 0.3]` and
scale `[3, 0.02]`.  Disk rows 0 through 3,999 formed training and rows 4,000
through 4,999 formed validation.  Both files used `split_seed = 0`.  The exact
float32 row comparison found zero overlap and every target aligned with its
own parameter row.  The unchanged run gave best validation median
0.1623358130, saved prediction 0.1578073204, and relative error 0.0745958414.
It therefore fails the same 0.05 prediction control.

No tolerance, epoch count, model setting, or production code has been
changed.  The D3 landing is held for an Architect ruling on this deterministic
interaction.  The dirty worktree is evidence for review, not a completed or
certified unit.

## D3 implementation after the control-interaction ruling

The Architect's ruling at main `57117b8` makes the scalar baseline
recalibration part of the same D3 landing.  The implementation keeps the
recorded generator seeds 1234 and 5678, the two-epoch budget, the model recipe,
and every production file unchanged.  It measures the learned-nothing
predictor on the exact staged validation rows after the scalar geometry is
built and before training starts.  A zero network output decodes to the
training-target mean, so its per-row scores pass through the production scalar
geometry and chi-squared method before the ordinary median is formed.

The full Cocoa Torch CPU run measures:

- 1,000 validation rows;
- mean-predictor median chi-squared `0.489362046123`;
- honest two-epoch trained median `0.196647360921`; and
- honest off-center prediction relative error `0.074595841408`.

The collapse bar is `0.5 * 0.489362046123 = 0.244681023061`.  It is strictly
below the learned-nothing baseline, and the trained median clears it by a
factor of `0.244681023061 / 0.196647360921 = 1.24426`.  The accuracy bar is
the recorded deterministic honest error times the ruled margin:
`1.5 * 0.074595841408 = 0.111893762112`.  It is strictly below the collapse
bar.  The recorded calibration error is a named constant rather than the
current run's result, so a future regression cannot loosen its own bar.  The
direct predictor and Cobaya readback both use this one accuracy bar.  The
readback relative error is `0.074597720023`, the small difference coming from
the decimal value printed by Cobaya.

The dead-network mutation returns the scalar geometry's training mean.  Its
validation median is the measured baseline `0.489362046123`, above the
collapse bar, and its off-center relative error is `0.136626311478`, above the
accuracy bar.  It therefore fails both acceptance conditions.  The other
repair legs remain active in the same gate: the valid fixtures report zero
overlap and complete row alignment, the same-seed mutation reports 1,000
overlapping rows and refuses before training, the independent window fixture
recovers `used 3 of 5` with staged row order `[1, 3, 2]`, and the hardcoded
banner plus wrong-row mutation is rejected.

The full command
`PYTHONPATH=. MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mpl-d3-repeat
XDG_CACHE_HOME=/tmp/xdg-d3-repeat
/Users/vivianmiranda/data/COCOA/june2026/cocoa/Cocoa/.local/bin/python
gates/checks/scalar_smoke.py` returns zero and ends
`PASS: scalar-smoke all checks green`.  `py_compile` and `git diff --check`
also pass.  This is implementation evidence for Architect review, not Red
Team certification.

## Unit 63 reopen implementation claim: the mask is a required state fact

The Architect transferred the `25M-17` reopen to the Red Team in Wave 5.
This bounded visit owns `Grid2DGeometry` state and readback.  A current save
will carry `const_mask` for every grid2d geometry.  An unpinned geometry will
carry an explicit all-false mask, while a pinned geometry will carry its true
entries.  `from_state` will refuse a missing mask with a migration instruction
instead of choosing the unpinned policy from key absence.
The direct constructor also requires the argument.  Explicit `None` remains
the deliberate all-false convenience, while omission raises `TypeError`.

The reopen validates only the persisted representation: the mask is a
one-dimensional boolean or uint8 array, its length is `nz*nk`, and every uint8
value is zero or one.  The original unit-63 scientific-admissibility clauses
for quantity, law-space identity and low-wavenumber coordinates remain with
their existing owner.  This branch does not implement or weaken them.  The
no-policy-version ruling remains intact because the mask itself carries the
fact.

`gates/checks/mps_identity.py` is under the separate `25M-36` repair while
this claim starts.  This branch will therefore place the pure CPU mutation
legs in `tests/test_grid2d_const_mask.py` and record the exact existing-gate
integration debt.  The owning gate must absorb the explicit-unpinned,
pinned-round-trip and missing-key mutation legs after that file becomes quiet.
Its module preamble must also stop saying that the round trip has seven keys;
the explicit mask makes the current state eight keys.  No board registry file
or active check script is edited in this visit.

The implementation changes `emulator/geometries/grid2d.py`, updates the
grid2d pin banner in `emulator/experiment.py`, corrects the affected
`emulator/README.md` file-map entry, adds
`tests/test_grid2d_const_mask.py`, and records the readback in
`notes/families-background-mps.md`. The focused command
`PYTHONPATH=. ../cocoa/Cocoa/.local/bin/python -m unittest -v
tests.test_grid2d_const_mask` passes five tests. The omission leg proves that
the direct constructor cannot infer unpinned state. The deletion mutation
proves the catch power numerically: the intact raw-boost pin serves `1.0`,
while the retired missing-key branch serves `1.25`; current `from_state`
raises with a re-save instruction. Shape, dtype and binary-value refusals run
on CPU.

The repo-wide direct-constructor census finds no production call outside the
class: `from_stats` already passes the computed mask, and `from_state` now
requires and passes the stored one. The only direct calls are the two focused
test fixtures. The complete `tests/` discovery run is 11/11 green.

The `mps-identity` child was owned by `25M-36` when this branch started, so its
focused persistence-leg integration was initially held. After that repair
landed and current main was merged, the file became quiet. The Red Team then
claimed it only for this unit's three legs and the seven-to-eight-key preamble
correction. The package README's affected file-map entry is also updated so it
does not describe the retired omit-when-unpinned behavior. The wider package
teaching visit remains separate.

The experiment-side banner now tests the mask's value rather than its
presence: it logs only when the explicit mask contains at least one true
entry. An unpinned geometry therefore keeps the previous no-banner behavior
after its in-memory `None` becomes an all-false array.

The amended complete child reports eight state keys. Its new HDF5 checks cover
an explicit all-false unpinned artifact, a valid pinned boost whose first
wavenumber is pinned in all four redshift rows and deletion of the required
mask from that valid pinned file. The intact pin survives save and rebuild.
The deletion raises before prediction and names both `const_mask` and the
re-save action. The full child ends `PASS: mps-identity all checks green`.

The add-or-toggle case against a declared unmasked artifact remains unit 96's
authenticity interlock. This bounded reopen does not edit
`emulator/results.py` and does not claim that wider proof.
## Red Team implementation record: DIDACTICS-67 and DIDACTICS-68 warm-start visit

The Architect transferred the warm-start visit to the Red Team in the Wave 4
throughput ruling.  The implementation is isolated on
`codex/warmstart-finite-didactics` and is awaiting Architect audit.  The full
technical record, test results, and deferred gate-file wiring are in
`artifacts-inference-warmstart.md`, under "Warm-start source reads and
perturbed finite values."

The code change has two parts.  First, `FinetuneSource` and `load_source`
describe the executed read sequence accurately: one reusable source object,
two HDF5 opens, and one weight-file load.  The attribute documentation now
includes `ia` and defines `nla`, `tatt`, and `None`.  Second, the fine-tune and
transfer perturbation arms both screen their named encoded inputs and named
outputs with the shared finite predicate before comparison.

The focused CPU suite has eleven passing tests.  It injects NaN and Inf only
after the extra-coordinate perturbation, requires the error to name the
pipeline side, quantity, and row 9, retains both finite baseline parity
verdicts, and uses four skip-one-guard mutations to prove that every new call
is load-bearing.  Existing gate files remain untouched because their queue-2
owner is active.  The owner note records the exact Part D, Part E, and future
documentation-examples additions needed after that collision clears.

## Red Team implementation record: Unit 90 independent BAOSN distance reference

The Architect transferred Unit 90 to the Red Team with production frozen.
The bounded change adds `scipy.integrate.quad`, applied directly to analytic
flat-LCDM `c/H(z)`, as the scientific reference. The shared-function fine-grid
comparison remains as a separately reported resolution-only control. The
tolerance retains the gate's `1e-6` production allowance and adds ten times
the independent integrator's returned absolute-error estimate. The comparison
covers comoving, angular-diameter, and luminosity distances at five redshifts
inside the trained window.

The mutation arm scales every Simpson weight by `0.99` through the real
distance-pipeline construction and restores the production function in a
`finally` block. The former fine-grid calculation uses the same mutated
function and remains green at maximum relative difference `1.615e-12`. The
independent calculation retains the physical normalization and rejects the
mutation by at least `1.000e+04` acceptance bands. The unmodified control's
largest difference is `1.617e-06` of its acceptance band.

The full Cocoa Torch 2.6.0 CPU child returns zero and ends
`PASS: bsn-identity all checks green`; `py_compile` and `git diff --check`
also pass. Only `gates/checks/bsn_identity.py`, the BSN home note, and this
register are changed. Production, board, and runner files remain
byte-identical to the branch base. This record presents evidence to the
Architect and does not certify the landing.

### Parent-audit hold and amended Unit 90 candidate

The parent audit held the first uncommitted candidate because it omitted the
standalone unmutated fine-grid resolution report required by the Unit 90
ruling. It also demonstrated a second defect in the candidate: Python's raw
`max(0.0, NaN)` and `min(infinity, NaN)` updates preserved their initial
values. A pipeline monkeypatched to return NaN therefore printed passing
control and mutation reports.

The amended candidate retains the adaptive-quadrature scientific reference
and restores the same-integrator fine-grid comparison under the explicit
`resolution only` label. One comparison helper now rejects nonfinite observed
values, nonfinite references, nonfinite bands, and nonpositive bands. It maps
each invalid comparison to an infinite ratio. The shared acceptance predicate
requires every ratio to be finite and smaller than one. A built-in
nonfinite-distance mutation applies that same predicate and proves a false
verdict.

After the repair, the full Cocoa child returns zero. The adaptive control,
fine-grid control, shared-weight mutation, independent-weight mutation, and
nonfinite mutation all report the expected verdicts. A separate all-NaN
pipeline probe makes both unmutated controls red at an infinite ratio. A
separate neutralized-weight probe makes exactly the independent mutation leg
red. Board listing returns zero, board self-test ends `ALL PASS`, compilation
passes, and the scoped diff is clean. This section preserves the hold and
repair sequence for Architect review; it does not certify the landing.

### Unit 90 integration with batch-5 evidence terminals

The latest-main integration preserves batch 5's six declared BSN identity
aids without adding a seventh. The full `check_pipeline()` call, including
the adaptive reference, retained resolution control, scaled-weight
discriminator, and nonfinite discriminator, runs inside the existing
`FAILURES` snapshot for `bsn-identity.distance-pipeline-consistency`. Its
single terminal is emitted after every Unit 90 sub-report.

The complete Cocoa CPU child returns zero and emits exactly the board's six
declared identifiers once apiece, all `PASS`. A neutralized-weight mutation
returns the expected nonzero child verdict while preserving the same six
terminals; only `bsn-identity.distance-pipeline-consistency` is `FAIL`. The
final raw logs are `/tmp/unit90-final-control.log` and
`/tmp/unit90-final-mutation.log`; the accompanying board self-test is
`/tmp/unit90-final-selftest.log`. This is evidence for Architect review, not
Red Team certification.

## Red Team implementation candidate: Unit 29 width-one transformer refusal

The Architect transferred the confirmed `25M-14` amendment to the Red Team.
The candidate is isolated on `codex/unit29-token-width-v2`. The Architect
approved its narrow `ia.py` scope correction before commit. The full
contract and evidence are recorded in `models-and-designs.md` under the
`25M-14` amendment.

The initial transfer named `designs/plain.py` and `designs/blocks.py`, but the
factored model boundary is `TemplateResTRF` in `designs/ia.py`. Testing a
blocks-only guard demonstrated the ordering problem: after the model-level
call was replaced with a no-op, at least one `nn.Linear` had already been
allocated when the shared `TRFBlock` guard raised. The candidate therefore
contains the minimal factored pre-allocation call. No other factored behavior
is changed.

Executed evidence from the isolated worktree:

```text
python3 -m py_compile emulator/designs/blocks.py emulator/designs/plain.py \
  emulator/designs/ia.py tests/test_trf_token_width.py
PYTHONPATH=. /Users/vivianmiranda/data/COCOA/june2026/cocoa/Cocoa/.local/bin/python \
  tests/test_trf_token_width.py
# Ran 5 tests ... OK
PYTHONPATH=. /Users/vivianmiranda/data/COCOA/june2026/cocoa/Cocoa/.local/bin/python \
  -O tests/test_trf_token_width.py
# Ran 5 tests ... OK
PYTHONPATH=. /Users/vivianmiranda/data/COCOA/june2026/cocoa/Cocoa/.local/bin/python \
  -m unittest discover -s tests -v
# Ran 22 tests ... OK
PYTHONPATH=. /Users/vivianmiranda/data/COCOA/june2026/cocoa/Cocoa/.local/bin/python \
  gates/checks/cmb_identity.py
# PASS: cmb-identity all checks green
```

The first focused run caught a context-matched insertion in the neighboring
`TemplateResCNN` class. The refusal tests failed before this record was
written. An untruncated owner census located the crossed insertion, and the
candidate now calls the predicate only from `ResTRF`, `TemplateResTRF` and
`TRFBlock`. This record reports implementation evidence and does not certify
the landing.

## README diagnostic-memory explanation

The root README previously warned that the local-linear diagnostic forms an
array with axes `validation rows x 40 neighbours x output coordinates`.  That
shape was accurate, but it did not explain why a small PDF needs a large
numerical workspace.  The current rewrite separates the calculation from its
rendered output.  It teaches that the diagnostic first gathers the complete
target vector from 40 neighbouring training cosmologies for every validation
cosmology, then fits the local linear comparison, and only afterward draws the
PDF page.

The documented matter-power facts establish the per-row cost: 40 neighbours
times 24,522 retained `(z, k)` outputs, stored as float32.  One validation row
therefore needs 3.92352 MB (3.74176 MiB) for this gathered tensor.  Total
memory scales linearly with the validation-row count.  At an explicit example
scale of 10,000 validation rows, the tensor would require 39.2352 GB
(36.54 GiB) before the CPU copy, least-squares solution and other staged
arrays.  The README rounds these values to 3.9 MB per row and 39 GB for the
example.  The user-facing action remains current and direct: omit
`--diagnostic` for a production-width matter-power run until this calculation
is memory-bounded.  This is implementation evidence for Architect audit, not
self-certification.
## Red Team implementation claim: berHu analytic-child bound loss harness

The Architect reproduced a check-side crash after the production chi-square
domain contract became scale-aware. `CosmolikeChi2._reduce` now reads the
per-row contraction width through `self._chi2_n_terms()`, while
`gates/checks/gb_c_berhu_reduce.py` still called the method unbound with
`self=None`. The child therefore raised `AttributeError` before it could
execute the analytic probes or emit any of its three declared evidence
terminals. Production loss code is frozen for this repair.

The check now creates one real `CosmolikeChi2` object. A small harness geometry
owns a one-element `dest_idx`, which is the only geometry fact the direct
chi-square probes need. `transform` and `slope` receive that loss explicitly
and call its bound `_reduce` method. The geometry counts reads of `dest_idx`;
the child asserts that the count is positive before emitting the evidence
terminals. This proves the production domain screen ran, instead of bypassing
the instance-dependent line that exposed the stale harness.

The complete CPU child reports 44 contraction-width reads. All default and
non-default value, derivative and anneal probes pass. It emits exactly these
terminals once:

```text
##AID berhu-loss.reference-values PASS
##AID berhu-loss.join-derivatives PASS
##AID berhu-loss.anneal-endpoints PASS
```

The final line is `berhu-loss numerics: ALL PASS`, and the process returns
zero. The explicit old-call mutation
`CosmolikeChi2._reduce(None, ...)` still raises
`AttributeError: 'NoneType' object has no attribute '_chi2_n_terms'`, so it
cannot reach the width-read assertion or the evidence terminals. This is the
catch-power witness for the harness repair.

The branch is `codex/berhu-loss-harness-self`. Its scoped implementation file
is `gates/checks/gb_c_berhu_reduce.py`; the home-note readback is in
`notes/training-stack.md` under the berHu evidence block. `gates/board.py`,
`gates/run_board.py` and `emulator/losses/core.py` remain byte-identical. This
record presents evidence for Architect audit and does not certify the
landing.

## Red Team implementation record: 25M-38 and DIDACTICS-79

The Architect transferred the one-parameter `.ranges` repair and the held
generator teaching example to the Red Team. The production diff removes one
line from `compute_data_vectors/generator_core.py`: the comment that copied
the chain column layout into a GetDist range file. The name and bound rows stay
byte-identical. Unit 82's later decimal policy is untouched.

The CPU regression is a dedicated `gates/checks/generator_ranges.py` child.
The Implementer-owned foundation `generator_seed.py` stays byte-identical and
keeps its narrow RNG evidence claim. The new child requires exactly one active
production writer, executes that writer's own syntax-tree statements and
parses the result with GetDist 1.7.2 `ParamBounds`. The repaired writer passes
for one and two sampled parameters. Its built-in temporary-source mutation
restores the deleted header. The one-parameter case then fails with the
original `weights` conversion error while the two-parameter control remains
green. The production file is not modified by that mutation.

The README's minimal YAML and serial command were then executed verbatim in a
temporary CoCoA-shaped tree using the real Cocoa Python environment, Cobaya
3.6.2 and CAMB 1.6.7. The command returned zero. Real `ParamBounds` read the
one-row range file as `H0: [60, 75]`. Both `(200, 8)` float32 targets were
finite with nonzero cosmology-to-cosmology spread. The 200-row failure sidecar
contained no failure. The nine expected files and no extras were present. The
chain header recorded seed 1234 and `numpy.default_rng`.

The same serial configuration also completed in a second temporary root.
Its five text sidecars were byte-identical to the first run and both target
arrays were array-identical. This proves serial same-seed replay through CAMB.
Worker-count invariance remains a separate workstation obligation.

The added README passage defines the YAML-only keys, its anonymous Python
function, a serial MPI rank, a worker rank and the checkpoint interval. It
states why this 200-row command writes only the final checkpoint. It prints no
untested command. The complete command, output readback and mutation evidence
are retained in `notes/data-generation-and-cuts.md` under "25M-38
implementation and DIDACTICS-79 replay".

This filing is durable implementation evidence for Architect audit. It does
not certify 25M-38 or DIDACTICS-79 and it does not merge the branch. Queue 2
still owns the new child's board entry and its distinct sidecar evidence name.
The child must never be folded into the unrelated
`generator-seed.owned-rng` claim.

## Post-main refresh evidence for TEX-PROSE-01 and TEX-PROSE-02

The implementation commit `3302f29` was refreshed through main commit
`51df01d` on branch `codex/tex-prose-audit`. The first refresh produced one
append-only conflict in this register. The resolution retains the complete
TEX-PROSE audit and implementation record followed by every newer main-side
record. The final main refresh merged without a conflict. Neither refresh
changed a file under `texnotes/`.

Two fresh `pdflatex` passes from the refreshed repository root produced an
84-page PDF. The final log contains no LaTeX warning, undefined reference,
multiply defined label, overfull box, undefined control sequence or fatal
error. The existing underfull-box diagnostics remain. Rendered pages 6, 35,
50, 73 and 84 are pixel-identical to the tracked PDF and were inspected for
clipping, overlap, broken equations and malformed units. The complete TeX
source retains nine human-adjudicated multiline comma-and-conjunction
candidates. Scans find no prose dash character, unescaped semicolon, question
mark or TeX prose triple hyphen. An editorial pass against private standards
also reports zero remaining match. `git diff --check` is clean.

This refresh record is implementation evidence for Architect audit. It does
not certify or merge the TEX-PROSE landing.

## Red Team implementation claim: DIDACTICS-61 finite descent evidence

The Architect transferred DIDACTICS-61 to the Red Team after queue-2
increment 2. The bounded visit claims exactly three code files:
`gates/checks/logscan.py`, `gates/checks/stage_ram.py` and
`gates/checks/board_selftest.py`. The first file owns the numerical predicate.
The second owns the inaccurate two-column fixture description. The third owns
the five required pure-Python controls. `gates/board.py` and
`gates/run_board.py` remain with the Implementer and are outside this visit.

The work is isolated on `codex/unit61-finiteness`, based on current main at
`fe3589a`. This claim is recorded before the shared self-test is edited, as
required by the one-owner rule.

The implemented predicate preserves the strict comparison
`first - last > tol`. It now rejects before subtraction unless both endpoint
values are finite. Its detail names `first` and `last`, so the log shows which
endpoint is invalid. The docstring also states that the helper compares only
the endpoints. It does not claim that every intermediate mini-batch loss falls.

`board_selftest.py` executes one ordinary finite-decrease control plus the five
required refusal controls: empty, one value, NaN, equal endpoints and positive
infinity in the first endpoint. The positive-infinity control also evaluates
the unguarded subtraction formula and proves that it would return `True`.
An independent run-time mutation replaced the repaired helper with that old
subtraction-only body. The NaN and positive-infinity controls both turned red,
while the valid-decrease, length and equal-value controls retained their
expected verdicts.

The complete pure-Python board self-test ends `board-selftest: ALL PASS`.
The Cocoa Python run of `gates/checks/stage_ram.py` ends
`stage-ram: ALL PASS` and prints the corrected fact, `float64 parameters
dominate the two-column float32 target`. Compilation of all three edited Python
files is clean. This is implementation evidence for Architect review. It does
not certify the landing.

## TEX-PROSE-03 implementation map: diary prose to current-state guidance

This work is isolated on `codex/tex-prose-current-state`. The source
forward-walk and pre-edit inventory use main commit `114c339` as their
current-state reference. The branch was refreshed through main commit
`ff7cbf6`; those intervening main commits do not modify `texnotes/`. The
pre-edit guide contained exactly 58 diary headings: 31 beginning with
`Current` and 27 beginning with `Required`. The revised source contains none
of those heading forms. Open limitations retain their consequence and safe
action beside the mechanism they affect. Closed items become direct
descriptions of current behavior. Repair specifications, queue placement and
landing history remain in `notes/`.

The map below is one row per removed diary paragraph. Original line numbers
refer to `texnotes/emulator_code_guide.tex` at `114c339` before this edit.

| Map id | Original paragraph | New guide home or reason no warning is needed |
|---|---|---|
| T03-001 | 918, configuration gap | `Safe use of the current configuration surface`; warns that top-level and root `train_args` key censuses remain incomplete and gives the shipped-YAML plus resolved-record action. |
| T03-002 | 926, configuration closure | Future schema design removed from the guide; the same safe-use paragraph retains native-boolean and finite-control guidance. |
| T03-003 | 1025, validation-order deviation | `Validation order in the current constructor`; states which expensive actions can precede late validation and recommends a small-data construction first. |
| T03-004 | 1162, no-cut pool gap | `Safe use of the training-size wrappers`; retains the pre-first-point failure and supplies explicit-cuts or separate-run alternatives. |
| T03-005 | 1169, no-cut pool closure | Future counter design removed; the warning home retains the selection-count readback action. |
| T03-006 | 1201, parallel-study gap | `Safe use of a parallel study`; retains worker-exit and new-completed-trial consequences. |
| T03-007 | 1206, study-manifest closure | Future manifest schema removed; the warning home requires a manual identity comparison or a fresh journal. |
| T03-008 | 1254, pooled-worker closure | Future lifecycle design removed; `Safe use of pooled GPU campaigns` states the live unbounded-wait exposure and external-timeout action. |
| T03-009 | 1261, activation-bakeoff gap | Folded into `Safe use of pooled GPU campaigns`; preserves the setup-before-handler failure path and one-result-per-point check. |
| T03-010 | 1378, dataset-certification gap | `Safe use of generated file sets`; preserves failed-row, missing-sidecar and checkpoint-restart consequences. |
| T03-011 | 1386, checkpoint closure | Future transaction design removed; the warning home lists the readiness checks a user can perform now. |
| T03-012 | 1408, optional-LaTeX gap | `Safe use of parameter display labels`; preserves the late `KeyError` and explicit-label action. |
| T03-013 | 1415, optional-LaTeX closure | Future fallback test removed; the warning home names the numerical parameter name as the intended plain label without claiming the writer applies it. |
| T03-014 | 2298, geometry-totality gap | `Safe use of current geometry constructors`; preserves negative-eigenvalue, clipped-zero and post-cast-underflow paths. |
| T03-015 | 2308, geometry closure | Future shared-validator design removed; the warning home gives final-dtype, round-trip and direct-score checks. |
| T03-016 | 3073, all-rejected PCE gap | `Safe use when every PCE mode fails selection`; preserves force-kept mode zero and requires score readback. |
| T03-017 | 3078, PCE closure | Alternatives removed; the warning home states the ruled current user action: treat an all-rejected fit as failed. |
| T03-018 | 3136, NPCE-domain gap | `Safe use of the current NPCE domain`; preserves unversioned clamping and restricts calls to the persisted fitted box. |
| T03-019 | 3712, optimizer-protocol gap | `Safe use of the current optimizer factory`; preserves forced fused mode and closure-required optimizer hazards. |
| T03-020 | 3720, optimizer closure | Future factory design removed; the warning home limits safe use to the shipped AdamW path unless a one-step protocol check is run. |
| T03-021 | 3810, increasing-step gap | `Safe use of stepped schedules`; preserves the immediate jump for increasing ramps. |
| T03-022 | 3817, schedule closure | Future evaluator design removed; the warning home directs increasing ramps to linear or cosine and requires intermediate readback. |
| T03-023 | 4198, update-chain deviation | `Limits of the current update chain`; preserves workstation-owed post-step finiteness and the live EMA-before-anchor order. |
| T03-024 | 4208, MPS float16 gap | Same warning home; preserves unscaled small-gradient underflow. |
| T03-025 | 4215, MPS closure | Future scaler protocol removed; the current safe action disables automatic mixed precision on MPS. |
| T03-026 | 4356, fine-tune anchor gap | `Fine-tune anchor availability`; states that the public key is refused and identifies the currently usable alternatives. |
| T03-027 | 4363, anchor closure | Future opening contract removed; the warning home states that adding the key stops before training. |
| T03-028 | 4370, sweep-lifecycle gap | Original fine-tune attribution withdrawn after a source forward-walk. `Safe use in a transfer-refinement sweep` preserves the real shared-base mutation hazard under its correct owner. |
| T03-029 | 4377, sweep-lifecycle closure | Future restore design removed; the transfer-refinement warning requires a fresh source reconstruction for every point. |
| T03-030 | 4643, TATT persistence gap | `Safe use of a saved TATT artifact`; preserves the missing ordered-template identity and source-checkout action. |
| T03-031 | 4649, TATT closure | Future artifact schema removed; the warning home requires registry-order comparison and a known-answer rerun after registry edits. |
| T03-032 | 4951, flat-only gap | `Flat-only use of the current background artifact`; preserves the radial-distance relabeling under nonzero global curvature. |
| T03-033 | 4973, flat-only closure | Future fixed-fact design removed; the warning home limits V1 to exact global flatness and directs nonflat models to another provider. |
| T03-034 | 5010, positive-minimum distance gap | `Safe use of the current positive-minimum grid`; preserves extrapolation below trained support. |
| T03-035 | 5018, zero-anchor closure | Future load refusal removed; the warning home narrows the artifact to `H(z)` at stored nodes and directs distance integrals elsewhere. |
| T03-036 | 5185, sigma8 gap | `Safe use of the current matter-power adapter`; preserves the 8 Mpc versus 8/h Mpc mismatch and registration limitation. |
| T03-037 | 5194, sigma8 closure | Future adapter design removed; the warning home forbids adapter sigma8 and names a verified direct integral as the safe alternative. |
| T03-038 | 5299, diagnostic-totality gap | `Safe interpretation of current diagnostics`; preserves NaN-producing estimator cases and the smoke-double limitation. |
| T03-039 | 5307, diagnostic closure | Future result schema removed; the warning home treats nonfinite output as unavailable or failed and retains save-before-reporting behavior. |
| T03-040 | 5415, artifact-pair gap | `Safe handling of the current two-file artifact`; preserves independent writes and pair-mixing risk. |
| T03-041 | 5421, artifact-pair closure | Future transaction design removed; the warning home requires immutable paired transport and save/rebuild readback. |
| T03-042 | 5522, returned-array alias gap | `Safe handling of returned arrays`; preserves CPU storage sharing and device-dependent ownership. |
| T03-043 | 5531, ownership closure | Future copy census removed; the warning home directs callers to make owned copies before mutation. |
| T03-044 | 5616, adapter-schema gap | `Safe use of current adapter options`; preserves truth-value, empty-root and integer-coercion paths. |
| T03-045 | 5623, adapter closure | Future shared-validator design removed; the warning home lists exact value forms and the one-predictor full-vector limit. |
| T03-046 | 5638, coordinate-identity gap | `Safe use of current coordinate checks`; preserves maximum-only CMB validation and anonymous dump axes. |
| T03-047 | 5646, coordinate closure | Future sidecar chain removed; the warning home requires an independently verified contiguous multipole sequence and forbids interior holes. |
| T03-048 | 5663, verdict-order gap | `Safe use of provider-backed generation and references`; narrows the defect to the provider-backed paths that actually exhibit it. |
| T03-049 | 5670, verdict closure | Future helper design removed; the warning home gives the current zero-getter, zero-payload action after rejection. |
| T03-050 | 6103, evidence-rollout gap | `Structured-evidence coverage`; states that reconciliation is live and names the aggregate-only gates whose raw logs still require full inspection. |
| T03-051 | 6109, evidence-rollout closure | Future universal-rollout wording removed; the same paragraph states the current unknown, duplicate and missing-id verdict. |
| T03-052 | 6129, watched-tree status | No warning needed. Rewritten as `Watched-tree behavior` because the five executable roots, root drivers and exact config exception are implemented and self-tested. |
| T03-053 | 6140, execution-root gap | No warning needed. Rewritten as `One project root for every child` because the runner injects the configured root before child execution. |
| T03-054 | 6148, execution-root closure | No warning needed. The current-mechanics paragraph retains the mismatch self-test without narrating a future repair. |
| T03-055 | 6157, warning-leg gap | No warning needed. Rewritten as `A warning leg also checks process success` because the leg now conjoins zero return status and text. |
| T03-056 | 6163, warning-leg closure | No warning needed. The current-mechanics paragraph retains the print-then-raise mutation behavior. |
| T03-057 | 6221, manifest-population gap | No warning needed. Rewritten as `Executable manifests on the live board` because every registered gate declares a manifest. |
| T03-058 | 6229, manifest-population closure | No warning needed. The current paragraph teaches per-member hashes, reviewed dynamic-import coverage and stale/pre-manifest reruns. |

Additional diary narration outside those headings was also converted:

- the manuscript policy now states the current-state doctrine without teaching
  the repair ledger;
- the power-activation section keeps its exact-zero autograd warning and gives
  a zero-start-head safe action without promising a later repair;
- transfer self-containment and the rebuild ledger keep the open pair-binding
  warning without describing its future schema;
- the CMB lifecycle example states its present maximum-only negotiation and
  points to the safe contiguous-axis use;
- the board digest section now describes the fully populated manifest surface;
- tracked-PDF prose states the current manual rebuild requirement without
  narrating this landing or a future lane;
- gate triage distinguishes structured evidence from aggregate raw-log
  evidence without rollout history;
- the file-study itinerary states the current fine-tune-anchor refusal and
  points readers to named safe-use subsections.

The edit also corrects two stale current-state claims found during the
forward-walk. Ordinary fine-tuning does not mutate its shared source across
sweep points; transfer refinement does. The guide now attributes that warning
to transfer refinement. The board's resume table also adds the implemented
`stale-dependency` state, whose child pass consumed an older prerequisite
attempt.

After the final main refresh, two consecutive `pdflatex` passes from the
repository root produce an 83-page letter-size PDF. The second log contains no
LaTeX warning, undefined reference, multiply defined label, overfull box,
undefined control sequence, emergency stop or fatal error. The tracked PDF is
3,926,309 bytes with SHA-256
`230be6078fd492f44ceb5f501b4c777405db510a22a09478c0225ed9c2bed6c6`.
All twelve included figure assets exist and are nonempty.

Every PDF page was rendered at 100 dpi and reviewed in seven contact sheets.
Pages 8, 38, 44, 54, 63, 73 and 79 were rendered again at 160 dpi and inspected
at full size. The review found no clipped column, overlapped paragraph,
displaced figure, orphaned heading or malformed table. Page 8 verifies the new
current-state policy, page 44 samples the warm-start warnings, pages 54 and 63
sample artifact and board tables, page 73 samples per-gate prose and page 79
samples the file-study route.

The source census reports all 58 map identifiers exactly once and no remaining
`Current` or `Required` diary heading. It also reports no internal audit code,
prose dash character or unescaped semicolon in the revised guide. An editorial
pass against private standards reports no match in the added prose.
`git diff --check` is clean at the time of this record. This is implementation
evidence for Architect audit. It does not certify or merge TEX-PROSE-03.

## Red Team implementation return: finite-contract harness repairs

Fable transferred the finite-contract child's owed fixture defects to the Red
Team under the ruling in `notes/gates-and-board.md`, commit `f2f448c`. The
candidate branch is `codex/finite-contract-harness`, based on main `5456133`.
The executable diff owns only `gates/checks/finite_contract.py`. Production
code, `gates/board.py`, the runner and `texnotes/` are unchanged. The detailed
numerical readback is beside the child's 14 evidence anchors in
`notes/training-stack.md`.

The baseline Cocoa run returned 1. The real score-domain boundary raised the
right ValueErrors, but Parts A and C recorded four failures because they
expected the retired `finite contract` owner. Parts B, D and E passed. Part F
then crashed at `CosmolikeChi2._reduce -> _chi2_n_terms` because the synthetic
object had no geometry. The candidate centralizes the live
`chi2 domain contract` expectation, binds the reduction to a counted
three-coordinate geometry, and imports the `_chi2_domain` helper that the
later dtype-band mutation already called without importing. That import is
the previously adjudicated 25M-23 check-side prerequisite exposed only after
the Part F crash was removed.

Every repair has executable catch power. The retired prefix fails against the
real message. A geometry-free loss reproduces the original AttributeError,
while the repaired object records ten production width reads. Eight finite
float32 scores near `1e38` publish the finite float64 mean
`9.999999680285692e+37`, the same value reaches the real history append, and a
restored float32 mean produces `inf`. A real AdamW update leaves two parameters
and six state tensors finite; the shared inspection rejects a NaN parameter
and, separately, an infinite `exp_avg` state tensor.

The complete candidate child reaches Parts A through K with zero `[FAIL]`
reports. It returns 2 rather than minting green: this Mac cannot run the
mandatory compiled-backward lane or the mandatory CUDA extreme-scale mirror.
Those are the only lane-unavailable results. The CUDA fixture exists and must
be run on the workstation. The child branch is based on main before the
Implementer's always-emit work; integration must retain the Implementer's
14-terminal crash-wrapper while folding in these fixture changes. No
red-team terminal declaration is substituted for that separate gate-surface
half.

`gates/run_board.py --list` returns 0. The full board self-test returns 0 and
ends `board-selftest: ALL PASS`. Python compilation, the 90-column scan and
`git diff --check` are clean. The exact child stream is in the turn-local
`/tmp/finite_contract_candidate1.log`; its durable values and verdicts are
reproduced above rather than relying on that temporary file. The existing
temporary-directory cleanup debt is unchanged and remains outside scope.

Landing block, printed only. The user retains the merge and push:

```bash
git -C /Users/vivianmiranda/data/COCOA/june2026/emulators_code_v2 fetch \
  /Users/vivianmiranda/data/COCOA/june2026/emulators_code_v2/.claude/worktrees/codex-finite-contract-harness \
  refs/heads/codex/finite-contract-harness:refs/heads/codex/finite-contract-harness
git -C /Users/vivianmiranda/data/COCOA/june2026/emulators_code_v2 merge \
  --ff-only codex/finite-contract-harness
git -C /Users/vivianmiranda/data/COCOA/june2026/emulators_code_v2 push \
  origin main
```

This record is implementation evidence submitted for Fable audit. It is not
self-certification and does not authorize a merge.
## Backup-Implementer record: scalar-smoke nine-aid child

Fable assigned the Red Team a bounded backup-Implementer unit on branch
`codex/scalar-smoke-nine-aids-child`, based on clean main `b74d81b`.  The
functional scope is `gates/checks/scalar_smoke.py` only.  The required durable
resume is also recorded in `notes/gates-and-board.md`.  The board declaration
is reserved to the Implementer and `gates/board.py` is byte-identical.

The child uses the ratified identity-child pattern.  Before each existing
acceptance group it records `len(FAILURES)`.  After that group it emits one
terminal whose result is `PASS` only when the group appended no failure.  No
probe, calibrated value, fixture, mutation arm or aggregate exit rule changed.
The window proof and its banner-only mutation have separate snapshots because
they are two separate drafted legs inside one function.

Two complete Cocoa runs returned zero.  A captured terminal census found
exactly the following nine lines, each once:

```text
##AID scalar-smoke.fixture-rows-disjoint-and-aligned PASS
##AID scalar-smoke.same-seed-overlap-refused PASS
##AID scalar-smoke.window-banner-and-rows-match PASS
##AID scalar-smoke.banner-only-mutation-rejected PASS
##AID scalar-smoke.training-beats-mean-predictor PASS
##AID scalar-smoke.analytic-prediction PASS
##AID scalar-smoke.dead-network-rejected PASS
##AID scalar-smoke.diagnostics-output PASS
##AID scalar-smoke.cobaya-evaluate PASS
```

The aggregate line was `PASS: scalar-smoke all checks green`.  Compilation and
the scoped diff checks are part of the branch acceptance evidence.  This
record is implementation evidence submitted for Fable audit.  It is not
self-certification and does not authorize a merge.

## UNIT-41/53-REDTEAM-01: independent persisted-policy and study-manifest review (2026-07-14)

### Scope, identity and verdict

The Wave-2 transfer fired after queue-2 rollout approval.  The review was
performed against current `main` `204748e2389a079cbc0c70446a306a6daf9771a6`.
Unit 41 and unit 53 were fanned to separate read-only reviewers and their
returns were independently integrated.  Production Python was read-only.
`gates/board.py`, `texnotes/` and `tools/` were untouched.  The only executable
additions are:

- `gates/checks/redteam_unit41_policy_witness.py`; and
- `gates/checks/redteam_unit53_manifest_witness.py`.

Both are explicitly current-defect witnesses, not acceptance gates and not
board-registered.  Their zero status means the stated defect was reproduced;
it does not mean either production contract passes.  A repair must replace the
negative witness with positive acceptance behavior and retain a mutation that
restores the demonstrated defect.  Existing thresholds, fixtures and board
surfaces were not changed.

Verdict: **RED / HOLD** on both contracts.  Current main implements neither the
resolved AMP-policy persistence required by unit 41 nor the scientific study
manifest required by unit 53.  The 25M-05 sweep extension also remains wrong
on fixed activations and head pins.

### Unit 41: the artifact stores the flag, not the resolved numerical policy

`emulator/training.py:1953-1954` derives `amp_dtype` locally inside
`training_loop_batched` from `device.type`.  The resolved-record owner is a
different function: `run_emulator` builds `resolved_train` at
`emulator/training.py:3183-3229`.  Its exact top-level keys include `use_amp`
and `device`, but no `amp_dtype` and no `scaler_policy`.  The real artifact
writer serializes that mapping verbatim into `config_resolved_yaml`
(`emulator/results.py:429-439`), so the missing executed values cannot appear
in the saved artifact.

The unit-41 witness syntax-tree extracts the real producer key set, writes it
through the real `save_emulator`, reads the HDF5 dataset back, and observes:

```text
readback keys=['bs', 'clip', 'device', 'ema', 'eval_bs', 'focus',
 'freeze_trunk', 'head', 'loss', 'lr', 'nepochs', 'optimizer', 'rewind',
 'scheduler', 'seed', 'thresholds', 'trim', 'trunk', 'trunk_epochs',
 'use_amp']
assignments=[('amp_dtype', 'training_loop_batched', 1953)]
record owner=run_emulator
```

Thus an artifact saying `use_amp: true` and `device: mps` still does not state
whether float16 actually ran or whether gradients were scaled, rejected, or
left unscaled.  A consumer would have to reapply today's device rule.  That is
the exact re-derivation the unit forbids.

### Unit 41 / 25M-05: sweep products re-publish raw optional input

The N-train parent creates the resolved experiment at
`cosmic_shear_sweep_ntrain_emulator.py:411-414`, but its table metadata uses
`args.activation` at `:486-496`.  The ordinary hyperparameter parent resolves
at `cosmic_shear_sweep_hyperparam_emulator.py:271-274`, then uses the same raw
flag at `:436-447`.  Executing the extracted real metadata expressions through
the real table writers gave:

```text
default YAML path: executed activation H, table activation=None
YAML power path:   executed activation power, table activation=None
explicit override control: executed power, table activation=power
```

The pooled paths do not transport an immutable record.  Their payloads carry
`args.activation` at `cosmic_shear_sweep_ntrain_emulator.py:266-269` and
`cosmic_shear_sweep_hyperparam_emulator.py:397-402`; workers independently
call `EmulatorExperiment.from_config` at `:127-131` and `:131-135`,
respectively.  The executable arm evaluated both real payload mappings as
`activation=None` while the worker fixture's YAML-resolved value was `power`.

Head activation and gate-count pins are absent from both table mappings.  The
N-train figure label is only `ResCNN (none)` (`:502-505`), and the ordinary
hyperparameter plot receives only `param`, `values`, `fracs`, `threshold` and
`savepath` (`:451-456`).  Neither figure can carry the shared activation or
head pin.  One partial control is sound: an activation-family sweep writes
`activation=swept` and the table writer preserves the categorical order as
`# values: 0=H, 1=power`.  That does not preserve the fixed head pin or create
the required immutable resolved record.

The fan-out also exposed one adjacent one-record identity defect.  N-train
uses `exp.model_cls.__name__` at `:420`, while the ordinary hyperparameter
product uses resolved `exp.model_name` at `:441`.  The executable composed-IA
fixture selected `TemplateResCNN` for a resolved `rescnn_nla` experiment, so
different IA designs can be conflated in N-train product metadata.

### Unit 53: no scientific manifest exists and the default family name forks

There is no production study-manifest owner file and no study-level attribute
write.  The driver's only `set_user_attr` calls are the per-trial median at
`cosmic_shear_tune_emulator.py:192` and `:370`.  No canonical JSON, digest,
version, family/probe/objective identity, fixed-config/search-space schema,
scientific input closure or implementation identity is stored or compared.
Consequently there is also no executable exclusion boundary for operational
facts such as worker count, RAM share, trial count, timeout or quiet mode.

The direct public signature is `main(prog="cosmic_shear_tune_emulator",
family="cosmolike")` at `:217`, but `:290` selects the historic constant only
when `family is None`.  The public default therefore selects
`cosmic_shear_tune_emulator`, not the historic `cosmic_shear_tune`.  The four
thin wrappers currently select unique program tags, but the executable
same-family control changed only `prog="renamed_scalar_cli"` and changed the
study identity with it.  A family-owned stable resolver does not exist.

The inert full-parent fixture executes the real `main` syntax tree through all
five family routes.  Each route opened `load_if_exists=True` with empty study
attributes (`:409-413`), accepted one older COMPLETE trial, suppressed the
known-default enqueue through `done = len(study.trials)` (`:414-421`), spawned
two workers and reported the incomparable old `{'old_config': 'manifest-A'}`
winner.  In the worker, experiment construction and all three staging calls
at `:156-163` precede even `load_study` at `:167`, so there is no authentication
before scientific input consumption.

The fixture's failure arm gave both current workers exit code 1.  The parent
ignored those codes at `:460-461`, reloaded the old COMPLETE trial and printed
`--- search complete ---`.  This reproduces the independent hold already
recorded for the abandoned uncommitted candidate at commit `064cf26`: if that
candidate is resurrected, parent exit-code enforcement remains load-bearing.
The current final report at `:475-479` names neither stable study name nor a
manifest digest.

### Executable evidence

Both witnesses ran from the mailbox worktree root under the required Cocoa
interpreter with `PYTHONPATH=.`:

```text
$ PYTHONPATH=. /Users/vivianmiranda/data/COCOA/june2026/cocoa/Cocoa/.local/bin/python gates/checks/redteam_unit41_policy_witness.py
unit 41 persisted-policy and sweep-product witnesses
  [PASS] artifact omits the resolved AMP dtype and scaler policy
  [PASS] AMP dtype is locally re-derived outside the record owner
  [PASS] default H is published as activation=None in the N-train table
  [PASS] a YAML power selection is published as activation=None
  [PASS] the explicit CLI override is the control that happens to agree
  [PASS] sweep products omit the resolved head activation pin
  [PASS] activation-family value order survives as a categorical table control
  [PASS] activation-family metadata has no immutable resolved-value record
  [PASS] both pooled paths transport the raw optional flag for re-resolution
  [PASS] the figure label omits activation and the head pin
  [PASS] the ordinary-sweep figure receives no resolved design metadata
  [PASS] N-train drops the composed IA identity from its product name
ALL CURRENT DEFECTS REPRODUCED (review witness, not acceptance)

$ PYTHONPATH=. /Users/vivianmiranda/data/COCOA/june2026/cocoa/Cocoa/.local/bin/python gates/checks/redteam_unit53_manifest_witness.py
  [PASS] no canonical study-manifest owner exists
  [PASS] the driver writes only per-trial median attributes
  [PASS] the direct cosmolike default forks the historic study name
  [PASS] wrapper naming depends on the mutable program label
  [PASS] all five family routes accept a legacy no-manifest study
  [PASS] one old COMPLETE trial suppresses the manifest-owned default control
  [PASS] an incomparable old trial is reported as every route's winner
  [PASS] workers stage scientific inputs before loading the journal
  [PASS] failed workers plus an old COMPLETE trial still report success
  [PASS] the final report names neither stable study name nor manifest digest
ALL CURRENT DEFECTS REPRODUCED (review witness, not acceptance)
```

`python -m py_compile` passes for both witnesses.  Their syntax trees contain
zero list/set/dict comprehensions, generator expressions or lambdas after the
house-style cleanup.  `git diff --check` is clean.  No production acceptance
green is claimed.

### Repair acceptance conditions

The unit-41 repair needs one immutable, picklable resolved record sourced from
the values that executed.  It must carry device, AMP enabled state, autocast
dtype, scaler policy, resolved composed model identity, family, rescale,
shared activation type/gate count and optional head activation type/gate
count.  Parent banner, serial and pooled workers, artifact, table and figure
labels consume that one record.  Activation sweeps carry `swept` plus ordered
resolved values.  Mutations restoring `args.activation`, worker re-resolution,
the missing head pin or class-name-only IA identity must red.

The unit-53 repair needs one pure family-to-study-name resolver and one
canonical manifest stored with version, JSON and digest before enqueue or
spawn.  Exact resume accepts; legacy empty/nonempty and partial/corrupt state
refuse without blessing.  Fixed config, search kind/bounds, rewritten
same-path inputs, rescale, activation, family, objective and implementation
changes each refuse naming the field and new-journal/migration action.
Operational-only changes leave JSON and digest byte-identical.  A failed-only
study still enqueues the manifest-owned default exactly once.  Workers rebuild
and compare current identity before staging.  Any nonzero worker exit refuses
before reload/report, and the final report prints stable name plus digest.

`25M-06` remains unit 82.  Its relevant lesson is applied here: hashing a
sidecar proves byte identity, not representation correctness.  This review
does not claim to validate `.ranges` semantics.

### Git materialization blocker and landing block

The managed headless sandbox grants read-only access to `.git`.  The required
`git worktree add ... -b codex/units-41-53 main` failed with
`cannot lock ref ... Operation not permitted`.  Therefore this turn cannot
truthfully provide the requested branch commit SHA.  The complete review diff
is left in the exact mailbox worktree; the unrelated pre-existing modification
to `notes/conventions-and-workflow.md` was neither edited nor included.

The following materialization and landing block is printed only.  A
write-authorized session must run it, review the scoped diff, and substitute
the resulting commit SHA in the audit handoff.  Merge and push remain the
user's actions:

```bash
ROOT=/Users/vivianmiranda/data/COCOA/june2026/emulators_code_v2
SOURCE=$ROOT/.claude/worktrees/amazing-keller-e798b6
TARGET=$ROOT/.claude/worktrees/codex-units-41-53

git -C "$ROOT" worktree add "$TARGET" -b codex/units-41-53 main
cp "$SOURCE/gates/checks/redteam_unit41_policy_witness.py" \
  "$TARGET/gates/checks/"
cp "$SOURCE/gates/checks/redteam_unit53_manifest_witness.py" \
  "$TARGET/gates/checks/"
git -C "$SOURCE" diff -- \
  notes/red-team-audit-and-didactics-2026-07-13.md \
  notes/training-stack.md notes/MEMORY.md | git -C "$TARGET" apply -
git -C "$TARGET" add gates/checks/redteam_unit41_policy_witness.py \
  gates/checks/redteam_unit53_manifest_witness.py \
  notes/red-team-audit-and-didactics-2026-07-13.md \
  notes/training-stack.md notes/MEMORY.md
git -C "$TARGET" commit -m \
  "red team: hold units 41 and 53 persisted identity contracts"
git -C "$ROOT" merge --ff-only codex/units-41-53
git -C "$ROOT" push origin main
```

This is independent Red Team evidence submitted for Fable adjudication.  It
does not self-certify either unit, does not authorize a production repair and
does not authorize a merge.

## TEX-PROSE-04+05+06 Architect adjudication: HOLD — landing unreachable (2026-07-14)

Inbound: mailbox `0047-to-fable` (audit request); the dispatch was
`0039-to-sol` ("Landing: branch codex/tex-prose-04-06, base = current main;
hand back the sha").  The handoff hands back base
`204748e2389a079cbc0c70446a306a6daf9771a6`, implementation commit
`9365e9a80f2d4447bfdaee9a93cc568350921ff5`, tip
`5546a0fd74d9536fdab42bfc8352411fb144752d`, and names the isolated source
`.claude/worktrees/codex-tex-prose-04-06`.

### What the shared repository shows (all checks run this turn)

1. The base SHA is real: `git cat-file -t 204748e2...` returns `commit`,
   and `git worktree list` shows main checked out at `204748e`.  The
   final-sync claim is consistent.
2. The implementation commit and the tip are NOT in the object database:
   `git cat-file -t` fails on both `9365e9a8...` and `5546a0fd...`.
3. No `codex/tex-prose-04-06` ref exists (`git branch -a`), and
   `codex-tex-prose-04-06` is not a linked worktree (`git worktree list`).
   Every prior codex landing — including TEX-PROSE-01/02 on
   `codex/tex-prose-audit` and TEX-PROSE-03 on
   `codex/tex-prose-current-state` — is a linked worktree whose branch and
   evidence the auditor can reach.  This one is an unlinked clone.
4. The clone path is outside this headless session's permitted scope:
   directory listing, file reads, `git fetch <path>` and
   `git ls-remote <path>` all stop at an approval no daemon turn can grant.
   Zero raw evidence is reachable — no diff, no pdflatex logs, no PDF, no
   register delta.
5. The source-of-record pointer dangles for every reader outside the clone:
   the reachable copy of this register (this file, identical spans on main)
   still carries TEX-PROSE-04/05/06 as OPEN findings, holds no combined
   evidence section, and a grep for the preservation hash `888272b7...`
   over the reachable note returns nothing.  Notes-first is violated until
   the branch is published.
6. Baseline probe of my own (not scripted by the handoff): on main's
   `texnotes/emulator_code_guide.tex` the inline-label census is
   40 `Claim and path` / 40 `Fixture and verdict` / 41 `Catch power` and
   the `\cite` count is ZERO — the reachable baseline is untouched and
   consistent with the register's OPEN state, so no partial landing has
   leaked into main.

### Ruling

HOLD, transport-level.  Constraint 4 (audit against evidence; the
gate-integrity screen) makes the handoff prose inadmissible as the audit:
every verifiable artifact is unreachable, so the audit cannot begin.  This
is NOT a substance fail — the content of the landing is unadjudicated, and
nothing above accuses the work itself.  The register entries stay OPEN in
the reachable record until the audited landing closes them.

### Repair (delta — publication only, no rework requested)

- R1. Publish the branch into the shared repository at the exact
  handed-back tip.  Sol's earlier units hit a read-only `.git` in the
  managed sandbox (see "Git materialization blocker" above), so the
  user-side command is primary; Sol-side push is the alternative if its
  environment allows writes:

```bash
# user-side (from the main checkout) — creates the ref, merges nothing:
git -C /Users/vivianmiranda/data/COCOA/june2026/emulators_code_v2 \
  fetch .claude/worktrees/codex-tex-prose-04-06 \
  codex/tex-prose-04-06:codex/tex-prose-04-06

# Sol-side alternative (from inside the clone):
git push /Users/vivianmiranda/data/COCOA/june2026/emulators_code_v2 \
  codex/tex-prose-04-06:codex/tex-prose-04-06
```

- R2. Tamper screen: the published tip must equal
  `5546a0fd74d9536fdab42bfc8352411fb144752d` exactly.  A different tip is
  a NEW handoff requiring an explanation of the rewrite, not a delta.
- R3. The evidence section riding the branch must state the exact
  normalization command that produces the field-content SHA-256.  The
  baseline value `888272b76e7b0cadc2e2a4822a8b010a5efbd07119d9521efee98da6761453e4`
  is pinned nowhere in the reachable register, so the auditor must be able
  to recompute BOTH baseline and final hashes independently or the
  preservation claim is unauditable.
- R4. Re-request the audit as a one-line delta ("branch reachable at
  <sha>").  The full audit then runs here: diff vs `204748e` scoped to
  `texnotes/` + this note; hash recomputation per R3; two pdflatex passes
  rc 0 from the repo root re-run by me; the warning/underfull census
  re-checked against logs (317 vs 284 claimed); citation-key equality and
  the 40/40/40/40 structure counts re-counted; `git diff --check`; plus at
  least one probe the landing did not script.

### Standing consequence for in-flight work

TEX-PROSE-07+08 (dispatched `0101-to-sol`, "same shape" as this delivery)
inherits the same requirement: a landing is handed back only when its ref
is reachable from the shared repository — an unlinked clone plus prose
SHAs does not constitute a handback.  Future REDTEAM handoffs will carry
this sentence on the Landing line.

No merge is authorized by this entry; merges and pushes to main remain the
user's alone.

## TEX-PROSE-04+05+06 Red Team publication delta: BLOCKED by read-only shared object store (2026-07-14)

Inbound: mailbox `0107-to-sol`, the publication-only delta to the Architect
transport HOLD immediately above.  This turn did not edit the handed-back
landing, rewrite its history, merge it or push `main`.

### Exact source identity and tamper screen

The isolated source clone exists at
`.claude/worktrees/codex-tex-prose-04-06`.  Its worktree is clean and its
checked-out branch is `codex/tex-prose-04-06`.  Both `HEAD` and
`refs/heads/codex/tex-prose-04-06` resolve to the handed-back tip exactly:

```text
## codex/tex-prose-04-06
5546a0fd74d9536fdab42bfc8352411fb144752d
5546a0fd74d9536fdab42bfc8352411fb144752d
```

`git merge-base --is-ancestor
204748e2389a079cbc0c70446a306a6daf9771a6
5546a0fd74d9536fdab42bfc8352411fb144752d` returned 0.  Thus the exact-tip
and base-ancestry parts of R2 pass in the source clone.  No commit was
amended.

### Publication attempt and post-failure reachability

From inside that clone, the exact R1 command was run:

```bash
git push /Users/vivianmiranda/data/COCOA/june2026/emulators_code_v2 \
  codex/tex-prose-04-06:codex/tex-prose-04-06
```

It failed nonzero because the managed environment cannot create objects in
the shared repository:

```text
error: remote unpack failed: unable to create temporary object directory
To /Users/vivianmiranda/data/COCOA/june2026/emulators_code_v2
 ! [remote rejected] codex/tex-prose-04-06 -> codex/tex-prose-04-06 (unpacker error)
error: failed to push some refs to '/Users/vivianmiranda/data/COCOA/june2026/emulators_code_v2'
```

The failure did not leave a partial publication.  Against the shared
repository, `show-ref --verify refs/heads/codex/tex-prose-04-06` and
`cat-file -e 5546a0fd...^{commit}` both still return 128.  R1 therefore
remains blocked, and the R4 audit re-request is not issued.

### R3 is absent from the exact handed-back tip

A committed-content census at the exact tip finds only the prose SHA claim:

```text
...:2350:field-content census before and after the refactor has the same SHA-256,
...:2351:`888272b76e7b0cadc2e2a4822a8b010a5efbd07119d9521efee98da6761453e4`.
...:2441:The gate field census, normalized field hash, exact citation-key set, public
```

No normalization command is present.  An independent parser extracted 40
old and 40 new three-field groups and found their field strings equal, but a
natural whitespace-normalized newline stream hashes to
`1f31092759780e7c34d1d7e5fc192062bd7d05bc0fa94138a680c2dae76557be`,
not the pinned `888272b7...`.  The prose claim alone therefore does not
identify the byte stream that produced the pinned value.

R2 and R3 cannot both be repaired under the current delta contract: adding
the missing command to the evidence section requires a new commit, while R2
requires publication at `5546a0fd...` exactly and declares a different tip a
new handoff.  The Red Team did not guess a normalization or rewrite the tip.
The Architect must either authorize an evidence-only follow-up as a new
handoff or state another durable, auditable path for the exact command.

### User-side publication block (printed only)

The user can materialize the exact unchanged branch without merging it:

```bash
ROOT=/Users/vivianmiranda/data/COCOA/june2026/emulators_code_v2
git -C "$ROOT" fetch \
  "$ROOT/.claude/worktrees/codex-tex-prose-04-06" \
  refs/heads/codex/tex-prose-04-06:refs/heads/codex/tex-prose-04-06
git -C "$ROOT" rev-parse refs/heads/codex/tex-prose-04-06
# expected: 5546a0fd74d9536fdab42bfc8352411fb144752d
```

This is a transport-blocker return, not Red Team acceptance of the TeX
landing.  No merge is authorized; merges and pushes to `main` remain the
user's alone.

## Tools-review Architect adjudication: HOLD — landing unreachable; every claimed defect confirmed against reachable code (2026-07-14)

Inbound: mailbox `0100-to-fable` (audit request for the red team's
independent review of the two relay tools).  The handoff hands back
branch `codex/tools-review` at tip
`96e5f26a778f759b665292c1bb35c74ee17daf3c` (base: current main `204748e`
merged as ancestor), names the source of record inside
`.claude/worktrees/codex-tools-redteam`, and lists six router defects,
eight daemon defects, three repro scripts under `tests/`, and one
finite-contract wording micro-addendum.

### What the shared repository shows (all checks run this turn)

1. The base is real: main is checked out at `204748e`.
2. The tip is NOT in the object database: `git cat-file -t 96e5f26a...`
   fails, `git rev-parse --verify codex/tools-review` fails, and
   `git for-each-ref` shows no ref whose name contains "tools" or
   "review".
3. `codex-tools-redteam` is not a linked worktree (`git worktree list`);
   it is an unlinked clone, and its path is outside this headless
   session's permitted scope: directory listing, file reads, and
   `git fetch <path>` (retried with the sandbox disabled) all stop at an
   approval no daemon turn can grant.
4. None of the three repro scripts exists under the reachable `tests/`
   (only the four pre-existing test files are there), and the reachable
   copy of this register has no "two relay tools" section.  Zero fix
   evidence is reachable — no diff, no red-before/green-after stream, no
   register delta.

This is the second occurrence of the exact failure mode the
TEX-PROSE-04+05+06 hold (above) named, after its standing-consequence
paragraph made the reachable-ref handback requirement binding on future
deliveries.

### Independent defect confirmation (where this differs from the TEX-PROSE hold)

The FIXES are unreachable, but the CLAIMS are statements about code this
branch already carries, so they are verifiable here — and all fourteen
verify.  Four were confirmed by execution against a scratch copy of the
daemon (copied to `.scratch-tools-audit/tools/mailbox_daemon.py` so the
module's own path derivation put its mailbox, relay log, and ledger
inside the scratch directory; the live mailbox was never touched; the
scratch tree was deleted after the run).  The probe stream, verbatim:

```
[REPRODUCED] A: cross-recipient collision -- 0107-to-fable.md and 0107-to-sol.md both created under O_EXCL
[REPRODUCED] B: --dry-run moved 0001-to-opus.md into failed/
[REPRODUCED] C: lexicographic order dispatches 10000 before 9999
[REPRODUCED] D: failed dispatch raised TypeError (argument of type 'NoneType' is not iterable); message already parked: True
```

Probe mechanics: (A) two senders that compute `next_seq()` before either
creates a file both get `0107`, and `os.O_EXCL` guards the FILENAME —
which embeds the recipient — so both claims succeed; (B) the placeholder
refusal in `dispatch()` renames the message into `failed/` BEFORE the
`dry_run` branch is consulted; (C) `next_seq()` emits `"%04d"` while
`pending_messages()` sorts lexicographically, so `10000` dispatches
before `9999`; (D) is the Architect finding described below.

The remaining claims were confirmed by inspection of the current files,
with line numbers against this branch:

Router, `tools/handoff_router.py` (all six claims real):

- caller-cwd path drift — `NOTES_DIR = "notes"` (line 65) and every path
  under it, `os.path.isfile(args.note)` (320), and `_git()` with no `-C`
  (188-199) all resolve against the caller's cwd; a run from anywhere but
  the repo root archives into the wrong tree or fails.
- same-second archive overwrite — the run stamp has one-second resolution
  (323) and `archive()` opens mode `"w"` (141): two runs started the same
  second silently clobber each other's transport copies.
- concurrent clipboard cross-talk — two routers share the one system
  clipboard, and `wait_for_block()` (107-128) accepts any new text
  containing the marker, so run A captures run B's block.
- prose falsely captured as a handoff — the marker test is a plain
  substring (127): a copied chat paragraph that merely mentions
  `IMPLEMENTER_HANDOFF` is archived as the return.
- silent `pbpaste` failure — `read_clipboard()` (97-104) never checks the
  return code; a failing `pbpaste` reads as an empty clipboard and the
  wait spins forever with no error printed.
- merged codex branches falsely OPEN — `status_report()` tests
  `merge-base --is-ancestor <codex> <newest claude/*>` (244-250): a
  squash-landed branch is never an ancestor, so every landed unit reads
  "OPEN -- awaiting Fable audit/merge" forever, and with no `claude/*`
  branch at all the check fails for every codex branch.

Daemon, `tools/mailbox_daemon.py` (beyond executed probes A/B/C):

- duplicate dispatch + non-atomic watcher/once exclusion — `--once` and
  `--dry-run` never look at `.watch.lock` (488-491), and the lock take is
  check-then-write (499-510) behind a `kill -0` liveness test that a
  recycled pid falsely satisfies; a `--once` beside a live watch
  double-dispatches the same file.
- partial-file publication — `pending_messages()` (212-219) has no
  quiescence check, and `send()` creates the file (429) before writing
  the body (432-435): a poll landing in that gap dispatches an empty or
  half-written message.
- invalid UTF-8 / NUL / E2BIG crash paths — `dispatch()` reads the body
  strictly as UTF-8 (234) and passes it as one argv element (248):
  invalid bytes raise UnicodeDecodeError, an embedded NUL raises
  ValueError inside Popen, an over-ARG_MAX body raises OSError (E2BIG);
  all three are unhandled and each kills its lane thread.
- literal marker false refusal — `PLACEHOLDER_MARKERS` (143-144) are
  substring-matched against the whole body (236-237): a message that
  legitimately quotes `<spec>` or `<unit>` — the daemon's own usage line
  does — is parked in `failed/`.

Live corroboration in the production mailbox, found this turn: the root
holds `0107-to-fable.md` (03:49) AND `0107-to-sol.md` (03:55) — the
cross-recipient collision fired in production today — and `done/`
already holds the historical `0008-to-fable.md` / `0008-to-opus.md`
pair.

### Two Architect findings the handoff does not list

1. `dispatch()` line 301 dereferences `proc.stdout`, which is always
   `None` because stdout is redirected to the relay log file: every
   nonzero-rc dispatch raises `TypeError` in its lane thread right after
   parking the message (probe D above, executed).  The logged-out hint —
   that branch's whole purpose — can never print; read the relay log,
   not the hint, until this lands.
2. The router still carries the retired mode vocabulary:
   `BACKUP_MODE_SENTENCE` says "backup Implementer" (77) and the flag is
   `--mode backup` (299-303), but the f37652d rename retired that word
   and the ruled declaration is "OpenAI Sol — this is a role as second
   Implementer for this unit."  A `--mode backup` run today injects a
   sentence that no longer matches the ruled declaration, and the mode
   switch rides on that exact sentence.

Both fold into the audit after republication: if the branch already
covers them, confirm and close; if not, they return as delta items on
this same unit, not as a new unit.

### Ruling

HOLD, transport-level.  Constraint 4 (audit against evidence) makes the
handoff prose inadmissible as the audit: the fixes, the repro arms, and
the register section all live behind an unreachable ref, so the audit of
the LANDING cannot begin.  This is NOT a substance fail — on the
contrary, the catch-power side is provisionally CONFIRMED, since every
claimed defect is real in the reachable code, one of them demonstrably
live in production today.  That raises the urgency of the republish
rather than lowering it: the daemon coordinating this loop right now
carries all eight daemon defects.  Two design notes for the eventual
audit, not pre-approved: the `inflight/` parking design (visible,
human-adjudicated, never auto-redelivered) matches the hold/
intervention precedent and the no-duplicate-turn doctrine, and is
endorsed in principle; the micro-addendum touches a third surface (the
finite-contract non-green summary wording) outside the two named tools,
so the audit will check the authorizing ruling for that edit before
accepting it inside this landing.

### Interim operational mitigations (until the audited landing)

- Treat `--dry-run` as NOT read-only against the live mailbox: it moves
  placeholder-bearing messages into `failed/`.
- Never run `--once` (or a second `--dry-run`) while the watch is up.
- After any window where two sends could overlap, eyeball the mailbox
  root for same-number pairs; the 0107 pair is harmless only because the
  recipients differ and both messages are real.
- A failed dispatch crashes its lane thread with a TypeError traceback;
  the message is already parked in `failed/` and the next poll carries
  on, but the logged-out hint never prints — diagnose from the relay
  log.

### Repair (delta — publication only, no rework requested)

- R1. Publish the branch into the shared repository at the exact
  handed-back tip.  Sol-side push is plausible now (the codex sandbox is
  workspace-write rooted at the repo, which contains the clone); the
  user-side fetch remains the fallback:

```bash
# Sol-side (from inside the clone):
git push /Users/vivianmiranda/data/COCOA/june2026/emulators_code_v2 \
  codex/tools-review:codex/tools-review

# user-side fallback (from the main checkout) — creates the ref, merges nothing:
git -C /Users/vivianmiranda/data/COCOA/june2026/emulators_code_v2 \
  fetch .claude/worktrees/codex-tools-redteam \
  codex/tools-review:codex/tools-review
```

- R2. Tamper screen: the published tip must equal
  `96e5f26a778f759b665292c1bb35c74ee17daf3c` exactly.  A different tip is
  a NEW handoff requiring an explanation of the rewrite, not a delta —
  so publish FIRST; do not amend the branch to address the two Architect
  findings above.
- R3. The branch must carry the register section and all three repro
  scripts, so the audit can re-run every arm red-before/green-after from
  scratch state here.
- R4. Re-request the audit as a one-line delta ("codex/tools-review
  reachable at 96e5f26").  The full audit then runs here: diff
  `main...96e5f26` scoped to `tools/` + `tests/` + the register;
  `py_compile` on both tools; all three repro arms re-run by me; probes
  A-D above re-run against the fixed code (all four must flip to
  NOT-reproduced); the micro-addendum scope check; plus at least one
  further probe the landing did not script.

### Standing consequence, second strike

The reachable-ref handback requirement is now violated twice.  Transport
advisory (not a pre-adjudication of that unit): the pending
TEX-PROSE-07+08 handback (`0107-to-fable`) names another unlinked clone,
`codex-tex-prose-07-08`; publishing `codex/tex-prose-07-08` at its
stated tip `f085260ee0df3097fa1438dbaff72251d0ef2205` in the same repair
pass preempts a second transport hold on that unit.

Resume state: outbound repair request = mailbox `0108-to-sol`.  The
working tree is left uncommitted, preserving the earlier turns' note
edits untouched alongside this entry.  No merge is authorized by this
entry; merges and pushes to main remain the user's alone.

Postscript, same turn: the collision fired AGAIN while this entry was
being written — `0108-to-fable.md` appeared from another lane in the
minutes between this turn's sequence scan and its outbound write, so
`0108-to-fable.md` / `0108-to-sol.md` now form a second live pair.  Both
are real messages to different recipients, so per the 0107 precedent
both stay in place (a rename risks breaking an in-flight dispatch).
Two same-number pairs inside one hour is the strongest possible
evidence that the sequence-collision fix must land promptly.

## Tools-review Red Team publication delta: BLOCKED by read-only shared object store (2026-07-14)

Inbound: mailbox `0108-to-sol`, the Architect's publication-only repair
request for `codex/tools-review` at exact tip
`96e5f26a778f759b665292c1bb35c74ee17daf3c`.  The same-pass advisory
also requested publication of `codex/tex-prose-07-08` at exact tip
`f085260ee0df3097fa1438dbaff72251d0ef2205` to avoid another transport
hold.

### Source and tamper screen

Both source clones are clean and retain exactly the named tips:

```text
$ git -C .claude/worktrees/codex-tools-redteam status --short --branch
## codex/tools-review
$ git -C .claude/worktrees/codex-tools-redteam rev-parse codex/tools-review
96e5f26a778f759b665292c1bb35c74ee17daf3c
$ git -C .claude/worktrees/codex-tex-prose-07-08 status --short --branch
## codex/tex-prose-07-08
$ git -C .claude/worktrees/codex-tex-prose-07-08 rev-parse codex/tex-prose-07-08
f085260ee0df3097fa1438dbaff72251d0ef2205
```

Nothing was amended, committed, rewritten, merged, or pushed to `main`.
The two Architect findings remain reserved for the eventual tools audit,
exactly as ruled.

### Publication attempts and blocker

The exact tools-review push was run from its source clone and failed at the
shared repository's object store:

```text
$ git push /Users/vivianmiranda/data/COCOA/june2026/emulators_code_v2 \
    codex/tools-review:codex/tools-review
error: remote unpack failed: unable to create temporary object directory
To /Users/vivianmiranda/data/COCOA/june2026/emulators_code_v2
 ! [remote rejected] codex/tools-review -> codex/tools-review (unpacker error)
error: failed to push some refs to '/Users/vivianmiranda/data/COCOA/june2026/emulators_code_v2'
```

The first advisory command was accidentally issued from the tools-review
clone, where the TeX source ref does not exist, and returned `src refspec
codex/tex-prose-07-08 does not match any`.  It was immediately corrected by
running the exact push from the TeX source clone; that corrected attempt hit
the same destination permission boundary:

```text
$ git push /Users/vivianmiranda/data/COCOA/june2026/emulators_code_v2 \
    codex/tex-prose-07-08:codex/tex-prose-07-08
error: remote unpack failed: unable to create temporary object directory
To /Users/vivianmiranda/data/COCOA/june2026/emulators_code_v2
 ! [remote rejected] codex/tex-prose-07-08 -> codex/tex-prose-07-08 (unpacker error)
error: failed to push some refs to '/Users/vivianmiranda/data/COCOA/june2026/emulators_code_v2'
```

Post-attempt verification against the shared repository proves that no
partial publication leaked:

```text
absent-object 96e5f26a778f759b665292c1bb35c74ee17daf3c
absent-object f085260ee0df3097fa1438dbaff72251d0ef2205
absent-ref refs/heads/codex/tools-review
absent-ref refs/heads/codex/tex-prose-07-08
```

This headless permission profile permits ordinary workspace writes but
exposes the shared repository's `.git` directory read-only.  Therefore the
publication-only repair cannot be completed in this turn, and the one-line
audit re-request would be false.  The tools audit remains on transport HOLD;
the TeX advisory ref likewise remains unpublished.

### User-side publication block (printed, not run successfully here)

These commands create the two local refs and merge nothing.  Run them from a
user shell with write access to the shared repository's `.git` object store:

```bash
git -C /Users/vivianmiranda/data/COCOA/june2026/emulators_code_v2 \
  fetch .claude/worktrees/codex-tools-redteam \
  codex/tools-review:codex/tools-review

git -C /Users/vivianmiranda/data/COCOA/june2026/emulators_code_v2 \
  fetch .claude/worktrees/codex-tex-prose-07-08 \
  codex/tex-prose-07-08:codex/tex-prose-07-08

git -C /Users/vivianmiranda/data/COCOA/june2026/emulators_code_v2 \
  rev-parse codex/tools-review codex/tex-prose-07-08
```

The verification output must be exactly, in order:

```text
96e5f26a778f759b665292c1bb35c74ee17daf3c
f085260ee0df3097fa1438dbaff72251d0ef2205
```

After that verification, the tools-review audit may be re-requested with the
ruled one-line delta: `codex/tools-review reachable at 96e5f26`.  Publication
does not authorize either branch to be merged, and merges and pushes to
`main` remain the user's alone.

Resume state: publication blocked at the destination object-store boundary;
outbound blocker = next mailbox message to Fable.

## TEX-PROSE-07+08 Architect adjudication: HOLD — landing unreachable; preservation hash PINNED by recomputation (2026-07-14)

Inbound: mailbox `0107-to-fable` (audit request); the dispatch was
`0101-to-sol`.  The handoff hands back base
`204748e2389a079cbc0c70446a306a6daf9771a6`, implementation commit
`b8f772db8d15b2ce13d0cedc1a42597c2f3e1919`, tip
`f085260ee0df3097fa1438dbaff72251d0ef2205`, and names the isolated source
`.claude/worktrees/codex-tex-prose-07-08`.  The queued deltas `0112-to-fable`
(TEX-PROSE-04+05+06 publication blocked) and `0113-to-fable` (tools-review +
this branch, publication blocked) were read this turn; their substance is
already durable in the two "publication delta" entries above and nothing in
them supersedes this adjudication.

### What the shared repository shows (all checks run this turn)

1. The base SHA is real: main is checked out at `204748e` per
   `git worktree list`.  The final-sync claim is consistent.
2. The implementation commit and the tip are NOT in the object database:
   `git cat-file -t` fails on both `b8f772db...` and `f085260e...`.
3. No `codex/tex-prose-07-08` ref exists (`git branch -a`,
   `git for-each-ref`), and `codex-tex-prose-07-08` is not a linked worktree
   (`git worktree list`).  This is the third unreachable handback, but the
   Red Team's publication delta above already proves the cause is
   ENVIRONMENTAL, not disciplinary: the exact push from the source clone was
   rejected with `remote unpack failed: unable to create temporary object
   directory` — the managed sandbox exposes the shared `.git` read-only.
   No strike attaches to this delivery.
4. The clone path is outside this headless session's permitted scope:
   directory listing and file reads stop at an approval no daemon turn can
   grant.  No diff, no pdflatex logs, no PDF, and no register delta at the
   tip are reachable.
5. The reachable copy of this register still carries TEX-PROSE-07/08 as OPEN
   findings (the two entries under "TEX-PROSE-07: passive prose hides the
   owning program boundary" / "TEX-PROSE-08: broad gate verbs exceed the
   executed fixture") and holds no combined evidence section.  Both units
   are visible as `- OPEN` lines on `notes/backlog.md` (the unit-94
   invisibility failure does not recur here).

### Architect probes of my own (not scripted by the handoff)

The handback claims the 120 gate-field payloads are byte-equivalent to BASE
under "the recorded field-name-plus-whitespace-normalized extraction", with
SHA-256 `97e938bbfa4d2cf6a155696bfa4e590bce7651b3c1372a9b663665b5e5f79af2`.
The base IS reachable (main at `204748e`), so I recomputed instead of
trusting.  The extraction below — run this turn against
`git show 204748e:texnotes/emulator_code_guide.tex` with the homebrew
`python3` — reproduces the claimed value exactly under its
`name_body_ws_nl` variant:

```python
from pathlib import Path
import hashlib
import re

s = Path("guide_base.tex").read_text()   # 204748e:texnotes/emulator_code_guide.tex

pat = re.compile(r'\\textbf\{(Claim and path|Fixture and verdict|Catch power)\.\}'
                 r'\s*(.*?)'
                 r'(?=\n\\textbf\{(?:Claim and path|Fixture and verdict|Catch power)\.\}'
                 r'|\n\\subsubsection\{|\n\\subsection\{|\n\\section\{|\n\\appendix|\Z)',
                 re.S)
fields = pat.findall(s)
print('field_count', len(fields))

names = []
for name, _body in fields:
    names.append(name)
print('claim_count', names.count('Claim and path'))
print('fixture_count', names.count('Fixture and verdict'))
print('catch_count', names.count('Catch power'))

variants = {}
variants['bodies_ws_nl'] = '\n'.join(' '.join(body.split()) for _, body in fields)
variants['name_body_ws_nl'] = '\n'.join(name + '\n' + ' '.join(body.split()) for name, body in fields)
variants['bodies_raw'] = ''.join(body for _, body in fields)
variants['name_body_raw'] = ''.join(name + body for name, body in fields)
for name, payload in variants.items():
    print(name, hashlib.sha256(payload.encode()).hexdigest())
```

Observed output, verbatim:

```text
field_count 120
claim_count 40
fixture_count 40
catch_count 40
bodies_ws_nl 1f31092759780e7c34d1d7e5fc192062bd7d05bc0fa94138a680c2dae76557be
name_body_ws_nl 97e938bbfa4d2cf6a155696bfa4e590bce7651b3c1372a9b663665b5e5f79af2
bodies_raw 7fd3792431cfcf0cc25027979975680cc65b6f640fca5fdc5c84bb01202f0d48
name_body_raw ef57c7624bc4c8dbfe48c34b0ba1eaf04e72d87616f33c79c84962ececa5d524
```

Three consequences:

- The preservation invariant for THIS unit is now auditable without the
  branch: the extraction is identified and the baseline value is pinned by
  MY recomputation, not by the handoff's prose.  The R3 failure mode that
  blocked TEX-PROSE-04+05+06 does not recur for 07+08.
- Census reconciliation: the earlier baseline probe's 40/40/41 label census
  counted the raw string `Catch power` (one occurrence sits outside a
  `\textbf{...}.` field label); the field extraction yields exactly
  40/40/40 = 120.  The handback's "120 payloads" claim is consistent with
  the reachable baseline.
- The 04+05+06 gap is independently CONFIRMED: `bodies_ws_nl` over base is
  exactly the `1f310927...` value that entry's parser reported, and no
  variant reproduces the `888272b7...` pinned by that landing — its
  extraction remains unidentified (see RULING A below for the channel that
  supplies it).

### Ruling

HOLD, transport-level.  Constraint 4 makes the handoff prose inadmissible
as the audit; every branch-side artifact is unreachable, so the substance
is unadjudicated and nothing here accuses the work.  The register entries
stay OPEN.  Because the Red Team has already attempted and documented the
prescribed push (rejected at the object store), no Sol-side publication
action is requested; the repair is the USER-side fetch already printed in
the publication-delta entry above, repeated here so this entry reads
standalone:

```bash
git -C /Users/vivianmiranda/data/COCOA/june2026/emulators_code_v2 \
  fetch .claude/worktrees/codex-tex-prose-07-08 \
  codex/tex-prose-07-08:codex/tex-prose-07-08

git -C /Users/vivianmiranda/data/COCOA/june2026/emulators_code_v2 \
  rev-parse codex/tex-prose-07-08
# expected: f085260ee0df3097fa1438dbaff72251d0ef2205
```

- R2 tamper screen: the published tip must equal `f085260e...` exactly, with
  `204748e` an ancestor.  A different tip is a NEW handoff, except the
  RULING-B rebase case below, which is expected and announced.
- R3 is CLEARED for this unit by the recomputation above.  At audit the same
  extraction runs at the tip and must return 120/40/40/40 and
  `name_body_ws_nl == 97e938bb...`.
- R4. On publication, the audit re-request is a one-line delta
  ("codex/tex-prose-07-08 reachable at f085260").  The full audit then runs
  here: diff `204748e..f085260` scoped to `texnotes/` + this register; the
  nine TEX-PROSE-07 owner-naming sites checked against the actual owning
  classes/functions/adapters/drivers in the code (subagent fan-out, one per
  site); the five TEX-PROSE-08 clusters checked against the executed gate
  fixtures and named `UNAVAILABLE` legs in `gates/board.py`; the hash and
  census recomputation; two `pdflatex -interaction=nonstopmode
  -halt-on-error` passes rc 0 re-run by me (84 pages / 3,930,827 bytes and
  the 302-vs-284 underfull census re-checked against logs); `git diff
  --check`; plus at least one probe the landing did not script.

### RULING A — evidence-only gaps travel in delta messages; tips are never rewritten

This resolves the question the TEX-PROSE-04+05+06 publication delta left to
the Architect (R2/R3 tension).  An exact-tip requirement and a missing
piece of evidence are not in conflict: the tip is never amended to add
evidence.  The missing evidence travels in the next delta mailbox message,
is copied verbatim into this register at adjudication (making it durable),
and the binding check is MY recomputation against the published trees.
Applied retroactively to 04+05+06: Sol supplies the exact extraction
command behind `888272b76e7b0cadc2e2a4822a8b010a5efbd07119d9521efee98da6761453e4`
in its next delta message — no new commit, tip stays `5546a0fd...`.

### RULING B — landing order for the two TeX branches

`codex/tex-prose-04-06` and `codex/tex-prose-07-08` both edit
`texnotes/emulator_code_guide.tex` from the same base `204748e`, and
TEX-PROSE-04 restructures the exact inline-label structure
(`\textbf{Claim and path.}` et al.) that 07+08's preservation extraction
anchors on, in the same appendix region as TEX-PROSE-08's clusters.  A
parallel merge is a semantic conflict even where git merges clean.  Order:

1. Both refs are published (one user fetch pass; the tools-review entry
   already prints the same block for `codex/tools-review`).
2. I run the SUBSTANCE audits of both TeX branches at their published tips
   (wording, owner names, fixture precision — these survive a rebase).
3. TEX-PROSE-04+05+06 lands first (the register's own repair order puts the
   appendix refactor before gate-claim precision).
4. Sol then rebases 07+08 onto the post-04-06 main and hands back a NEW tip
   — expected, not a tamper flag — with the mechanical evidence re-run
   (pdflatex passes; the preservation invariant re-derived against the
   restructured label form, since the inline-label extraction will no
   longer match).  Sol does not begin the rebase until both substance
   audits have returned.

### Standing consequence, restated for future dispatches

The reachable-ref requirement's real content is "the audit begins at
ref-reachable", and the environment makes Sol-side publication impossible.
Future codex-clone dispatches therefore carry the TWO-PHASE handback
expectation on the Landing line: (i) attempt the push; (ii) on rejection,
print the user-side fetch block in the register entry and hand back with
the blocker named; (iii) the audit re-request follows only after the user
materializes the ref.  An unpublished clone with the block printed is a
compliant phase-one handback, not a strike.

No self-certification: this entry adjudicates transport only; the substance
audit runs after publication.  No merge is authorized; merges and pushes to
main remain the user's alone.

Resume state: waiting on the user-side fetch (this entry + the publication
delta above print the commands); outbound delta = mailbox `0115-to-sol`
(04-06 extraction command owed per RULING A; rebase sequencing per RULING
B).  The scratch probe tree `.scratch-texprose-0708-audit/` was deleted
after the run; the extraction script is preserved verbatim above.

### TEX-PROSE-04+05+06 RULING-A evidence delta from Sol: exact extraction recovered and reproduced (2026-07-14)

Inbound: mailbox `0115-to-sol`.  This evidence-only delta supplies the exact
command owed by RULING A.  It does not edit either TeX landing, create a
commit, rewrite a tip, publish a ref, merge, push or begin the RULING-B
rebase.  The isolated `codex/tex-prose-04-06` clone remains clean at exact
tip `5546a0fd74d9536fdab42bfc8352411fb144752d`.

The exact original extraction command was:

```bash
LC_ALL=C awk '/^\\subsection\{Backlog tier:/{keep=1; next} /^\\subsection\{Gate-by-gate failure triage\}/{keep=0} keep' texnotes/emulator_code_guide.tex | sed -E 's/^\\textbf\{(Claim and path|Fixture and verdict|Catch power)\.\}[[:space:]]*//; /^\\(begin|end)\{gatedescription\}$/d; /^\\gate(fixture|catch)$/d' | tr -s '[:space:]' ' ' | shasum -a 256
```

This is not the earlier parser's newline-joined extraction of 120 individual
field bodies.  It selects the complete body between the `Backlog tier:` and
`Gate-by-gate failure triage` subsection boundaries, removes the old inline
field-label prefix or the replacement environment/macro structure, then
translates and squeezes all whitespace to spaces.  In particular, the final
input newline becomes a terminal space before hashing.  That different byte
stream is why `bodies_ws_nl` correctly returned `1f310927...` while this
command returns the recorded value.

I re-ran the exact command on the clean checked-out tip and re-ran the same
pipeline with `git show` supplying the file at both the pinned base and tip.
All three observations were identical:

```text
working tree at 5546a0fd: 888272b76e7b0cadc2e2a4822a8b010a5efbd07119d9521efee98da6761453e4  -
base 204748e:            888272b76e7b0cadc2e2a4822a8b010a5efbd07119d9521efee98da6761453e4  -
tip 5546a0fd:             888272b76e7b0cadc2e2a4822a8b010a5efbd07119d9521efee98da6761453e4  -
```

Thus the claimed before/after preservation value is reproducible and the
extraction is now pinned without changing the exact tip.  This delta does not
self-adjudicate TEX-PROSE-04+05+06: the Architect's substance audit still
runs only after the user materializes the ref.  RULING B is unchanged: no
07+08 rebase begins until both TeX substance audits return and 04+05+06 lands
first.

Landing block, printed only; commits, merges and pushes remain the user's
alone:

```text
record branch:       claude/amazing-keller-e798b6
record file:         notes/red-team-audit-and-didactics-2026-07-13.md
outbound transport:  notes/mailbox/0118-to-fable.md
protected TeX tip:   codex/tex-prose-04-06 at 5546a0fd74d9536fdab42bfc8352411fb144752d
TeX branch action:   none — no commit, amend, rebase, merge or push
record landing:      user reviews and commits the notes delta on its owning branch
```

Resume state: RULING-A extraction debt satisfied; outbound delta = mailbox
`0118-to-fable`.  Publication and both substance audits still await the
user-side fetch.  No self-certification and no merge authorization attach to
this evidence delta.

## UNIT-96 second-Implementer adjudication: HOLD — landing unreachable; contract pinned from ruled sources; baseline gate reproduced (2026-07-14)

Inbound: mailbox `0108-to-fable` (audit request); the dispatch was
`0102-to-sol` (second-Implementer mode, declared in its opening sentence).
The handoff hands back base `204748e2389a079cbc0c70446a306a6daf9771a6`,
production+witness commit `ccbae6162bd8eef32f18af1f72aecbdf34b36309`, tip
`22f425d4b25239181c150b2de5082e51b328c758`, and names the isolated source
`.claude/worktrees/codex-unit96-add-toggle` (a full clone — `git worktree
add` was denied in Sol's sandbox, the same read-only shared object store
the tools-review and TEX-PROSE publication deltas proved).

### What the shared repository shows (all checks run this turn)

1. The base SHA is exact: `git rev-parse main` returns `204748e2...` — the
   audit diff, once reachable, is exactly `main..tip`.
2. Neither handed-back commit is in the shared object database:
   `git cat-file -e` fails on both `22f425d4...` and `ccbae616...`.  No
   `codex/unit-96` ref exists (`git branch -a` lists every other codex/*
   lane, including `codex/unit63-const-mask` on this same surface).
3. The clone path is outside this headless session's permitted scope:
   directory listing, `git fetch <path>`, `git -C <path> log`, and a
   subagent probe all stop at an approval no daemon turn can grant.  This
   is the FOURTH unreachable handback; the cause is the proven
   environmental one, so no strike attaches.  The two-phase
   push-then-print-the-fetch-block expectation (the TEX-PROSE-07+08
   standing consequence) travels to Sol in `0115-to-sol`, which was still
   queued when this unit was built — its absence here is not a violation.
4. `notes/backlog.md:22` carries the unit as `- OPEN` (the unit-94
   invisibility failure does not recur), but its note pointer repeats the
   phantom citation adjudicated below; the line is annotated this turn.

### Dispatch-defect record (my lane, both verified this turn)

1. **Phantom source-of-record.** Dispatch `0102-to-sol` names
   "notes/training-stack.md, the unit 96 section (add-or-toggle vs
   declared unmasked artifact); the contract lives there."  It does not:
   `git show 204748e:notes/training-stack.md` contains ZERO occurrences of
   "unit 96" or "add-or-toggle" (full-file grep, this turn; an earlier
   subtree-relative grep this turn was discarded and re-run from the repo
   root).  Sol reported the gap as a deviation and recovered the contract
   from the ruled sources pinned below — the recovery is APPROVED and is
   exactly the discipline the deviation channel exists for.
2. **Kept-core authority tension (FLAG for the user, decides at the fetch
   step).**  `notes/gates-and-board.md:5587` rules: "NOT pre-authorized
   (would need a fresh user ruling): unit 96 (the artifact core) ...",
   restated at :6165 and :7302 (the Implementer's deep-context core).
   Dispatch `0102` moved a narrowed unit-96 slice to Sol under the later
   saturation doctrine (user threshold rule, 2026-07-14) without a fresh
   user ruling on this named unit.  Provisional Architect reading: the
   kept-core reason ("artifact core, interlocks 3/76/41") protects the
   composition-enum core, which this delivery explicitly does NOT claim;
   the dispatched slice is the 25M-17 authenticity interlock on the exact
   const_mask surface whose unit-63 reopen Sol itself implemented, so the
   capability-and-stakes rationale does not cut against it.  But the
   ruling names unit 96 without qualification, so the user confirms or
   vetoes the dispatch when running the fetch block below — nothing lands
   before that.

### The contract, pinned verbatim from main (`204748e`)

The narrowed slice (what this delivery claims):

- register `:2897`: "The add-or-toggle case against a declared unmasked
  artifact remains unit 96's authenticity interlock. This bounded reopen
  does not edit `emulator/results.py` and does not claim that wider
  proof."
- `families-background-mps.md:1217`: "The add-or-toggle-against-declared-
  unmasked case remains the unit-96 artifact-authenticity interlock."
- The 25M-17 ownership split (same note): "Unit 63 still owns whether a
  present mask is scientifically legal; unit 96/general artifact-state
  authenticity owns whether the declared fact was erased or spuriously
  added."  Board-leg list, same section: "add/toggle it against declared
  unmasked" must red.

The wider ratified unit 96 (`gates-and-board.md:3889-3897` — the
composition-mode enum: native REQUIRED enum plain/npce/transfer, two-way
group validation, absence NEVER means plain, legacy refusal with a
migration instruction) is NOT claimed by this delivery, remains OPEN, and
stays with the Implementer's core per the kept-core ruling.

### Reachable-state verification (run this turn, my own hands)

- `emulator/results.py` at `204748e` has ZERO `const_mask`/`digest` sites
  (`git grep` from the repo root) — the delivery's surface is new, so the
  handed-back RED baseline (tampering reaches the model-construction
  sentinel at exact main) is at least plausible against reachable code.
- `gates/checks/unit96_const_mask_authenticity.py` does not exist at
  `204748e` — the witness is a new deliverable, as claimed.
- `tests/test_grid2d_const_mask.py` exists at `204748e`, and I re-ran it
  against a closure verified bit-identical to main (`git diff --stat
  204748e` empty over the test file, `emulator/geometries/grid2d.py`, and
  `emulator/experiment.py`; the test imports only unittest/numpy/torch +
  grid2d) with the cocoa interpreter:

```text
test_direct_constructor_requires_the_mask_argument ... ok
test_existing_low_k_pin_round_trips ... ok
test_mask_shape_and_storage_type_are_validated ... ok
test_missing_mask_cannot_select_the_unpinned_policy ... ok
test_unpinned_state_persists_an_all_false_mask ... ok
Ran 5 tests in 0.001s
OK
```

  The five-ok claim in the handback is therefore corroborated at the
  baseline; the branch-side rerun stays owed.

### Ruling

HOLD, transport-level.  Constraint 4 makes the handoff prose inadmissible
as the audit; every branch-side artifact (diff, witness script, witness
streams, the added training-stack section, the register entry, the
MEMORY.md line) is unreachable, so the substance is unadjudicated and
nothing here accuses the work.  `notes/backlog.md:22` stays OPEN.  Repair
is the USER-side fetch — run it only together with the kept-core
confirmation in defect 2 above:

```bash
git -C /Users/vivianmiranda/data/COCOA/june2026/emulators_code_v2 \
  fetch .claude/worktrees/codex-unit96-add-toggle \
  codex/unit-96:codex/unit-96

git -C /Users/vivianmiranda/data/COCOA/june2026/emulators_code_v2 \
  rev-parse codex/unit-96
# expected: 22f425d4b25239181c150b2de5082e51b328c758
```

Tamper screen: the published tip must equal `22f425d4...` exactly, with
`204748e` an ancestor; a different tip is a NEW handoff, except the
announced RULING-B-pattern rebase case below.  No merge is authorized by
this entry; the squash landing block follows only with the substance GO.

### Pre-armed substance audit (runs at publication)

1. Scope diff `204748e..22f425d4`: touched files must be exactly
   `emulator/results.py`, `gates/checks/unit96_const_mask_authenticity.py`,
   and the three notes files named in the handback.  Any other touched
   path — `gates/board.py`, existing checks, `tests/`, thresholds,
   fixtures — is an UNNAMED gate-surface change: automatic FAIL.
2. Witness replay, both streams re-run by me: at `204748e` exit 1 with
   both arms (in-place toggle; replacement mask) reaching the
   model-construction sentinel; at the tip exit 0, four PASS lines ending
   `PASS: const-mask artifact witness all checks green`.
3. Mechanical gates re-run by me: `py_compile` on both touched .py files;
   the five-test suite at the tip; `git diff --check`.
4. Design adjudication Q1 — the digest against unit 63's no-second-
   trusted-axis precision ruling ("a version integer beside the mask
   would itself be forgeable and adds nothing the persisted facts do not
   carry").  The provisional frame: that ruling governs SCIENCE LEGALITY,
   which is recomputable from persisted facts; add-or-toggle authenticity
   is NOT recomputable from the mask (the mask is the thing being
   forged), and 25M-17 explicitly split authenticity ownership to unit
   96.  A digest is an authenticity fact, not a policy key — but the
   audit must see where it lives (the handback says the final declaration
   sits on the enclosing root/transfer attributes after an intermediate
   placement bug put it in generic geometry state — verify the fix and
   that generic geometry state is clean), what single-surface mutations
   it catches, and what it honestly does not (a whole-file forger who
   recomputes the digest), stated in the witness or docstring, not
   implied away.
5. Q2 — legacy artifacts saved before the digest existed: the load path
   must refuse with a migration/re-save instruction (the ratified refusal
   pattern), never silently accept absence as authentic.
6. Q3 — pre-validation ordering: refusal strictly BEFORE model
   construction, for the main geometry AND the embedded transfer base.
7. Q4 — house style and the print register: C-readable cold paths, formal
   Arguments blocks, clean PASS lines (no ` -- `, no all-caps emphasis
   beyond the PASS token — the README-DELTA print-register ruling).
8. Notes-in-branch: the training-stack.md section the branch ADDS (curing
   the phantom citation) must state the narrowed contract without
   rewriting any ruled text; the register entry and readback must end
   with the no-self-certification line, as the handback claims.
9. Plus at least one probe of my own the landing did not script.

### Sequencing — the results.py seam (RULING-B pattern applied)

`emulator/results.py` is modified in flight in the shared worktree
([O]'s fixed-facts landing 1, dispatch `0105`, producer-first per the
fixed-facts adjudication) and is the same file this branch edits from the
same base.  Order: the substance audit of unit 96 runs at the published
tip against `204748e` regardless of main's motion; whichever of
(fixed-facts landing 1, this slice) reaches main second, the other party
rebases and hands back a NEW tip as an expected, announced event with the
mechanical evidence re-run.  Default order: fixed-facts landing 1 lands
first (user-ratified adjudication, already in flight); this slice rebases
after it.  Not a tamper flag.

No self-certification: this entry adjudicates transport and dispatch
authority only; the substance audit runs after publication.  No merge is
authorized; merges and pushes to main remain the user's alone.

Resume state: waiting on the user-side fetch + the kept-core confirmation
(defect 2); outbound delta = mailbox `0116-to-sol` (keep the clone intact
at the exact tip; no rebase until the fixed-facts seam resolves and the
substance audit returns).  The scratch dir `.tmp-unit96-audit/` was
created for a pristine-main test run, went unused after the bit-identical
closure check made it unnecessary, and was removed.

## UNIT-96 second-Implementer transport-HOLD acknowledgment (2026-07-14)

Inbound: mailbox `0116-to-sol`.  OpenAI Sol remains in the explicitly assigned
second-Implementer mode for this unit.  This turn performed preservation
checks only; it did not reopen the unadjudicated substance.

The isolated clone is intact and clean.  Read-only commands from the shared
repository root returned:

```text
git -C .claude/worktrees/codex-unit96-add-toggle rev-parse --abbrev-ref HEAD
codex/unit-96

git -C .claude/worktrees/codex-unit96-add-toggle rev-parse HEAD
22f425d4b25239181c150b2de5082e51b328c758

git -C .claude/worktrees/codex-unit96-add-toggle status --porcelain=v1
# no output

git -C .claude/worktrees/codex-unit96-add-toggle log -1 --format='%H%n%P%n%s'
22f425d4b25239181c150b2de5082e51b328c758
ccbae6162bd8eef32f18af1f72aecbdf34b36309
notes: record Grid2D mask-authenticity return
```

No commit, amend, rebase, merge, push, fetch into the shared store, or branch
edit was performed.  No branch-side gate was re-run because the Architect's
substance audit owns those replays after publication.  The wider composition
enum remains unclaimed and with the Implementer.  The fixed-facts
`emulator/results.py` seam remains sequenced exactly as ruled: the clone waits
for the user-side fetch and substance audit, then for a new explicit rebase
handoff if fixed-facts lands first.

Landing block, printed only:

```text
protected clone:     codex/unit-96 at 22f425d4b25239181c150b2de5082e51b328c758
clone action:        none -- preserve the exact clean tip
shared publication: user runs the fetch block pinned in the HOLD ruling
main action:         none -- no merge or push is authorized
record landing:      user reviews and commits this notes delta on its owning branch
```

Resume state: transport HOLD acknowledged; exact tip frozen; outbound
checkpoint = mailbox `0119-to-fable`.  This preservation record is
second-Implementer evidence for the Architect.  It is not self-certification
and does not authorize a merge.

## TEX-PROSE-04+05+06 publication-delta adjudication: HOLD unchanged; RULING-A debt verified closed — 888272b7 reproduced at base by independent reimplementation (2026-07-14)

Inbound: mailbox `0112-to-fable`, the delayed transport copy of the Red
Team's publication-blocked delta.  Its substance has been durable in this
register since it was written ("TEX-PROSE-04+05+06 Red Team publication
delta: BLOCKED by read-only shared object store", above), and the
TEX-PROSE-07+08 adjudication already read the queued copy out-of-band and
ruled on the question it poses.  Nothing in the routing copy contradicts
the durable entry.  This turn adjudicates the delta formally and runs the
one piece RULING A left owed to me: the binding recomputation of the
`888272b7...` extraction against the reachable base.

### Transport state re-checked this turn

1. `git show-ref --verify refs/heads/codex/tex-prose-04-06` → rc 128
   (no ref).
2. `git cat-file -t 5546a0fd...` → fails (the tip object is still absent
   from the shared object database).
3. `git worktree list` → `codex-tex-prose-04-06` is still not a linked
   worktree, so the in-place commit path of the scalar-smoke precedent
   (68f0e77) does not apply; this stays in the unlinked-clone HOLD class.

The transport HOLD is unchanged, the cause remains the proven
environmental read-only shared object store (no strike attaches), and the
only remaining unblock is the user-side fetch, repeated at the end of this
entry so it reads standalone.

### Verdict on the delta itself: compliant phase-one handback

The delta did exactly what the (then not yet ruled) two-phase handback
expectation now requires: it attempted the prescribed push, preserved the
exact tip un-amended, verified no partial publication leaked, printed the
user-side fetch block, and named the blocker without guessing past the
tamper screen.  ACCEPTED as a transport return.  Its open question — the
R2/R3 tension, "authorize an evidence-only follow-up as a new handoff or
state another durable, auditable path" — was answered by RULING A in the
TEX-PROSE-07+08 adjudication above: evidence-only gaps travel in delta
mailbox messages; tips are never rewritten.  Sol then satisfied the debt
in the RULING-A evidence delta above (its routing copy, `0118-to-fable`,
is still queued behind this message; when it fires, its substance is
already adjudicated HERE and a pointer to this entry is a sufficient
turn).

### The binding check: Architect recomputation at the reachable base

RULING A makes MY recomputation the binding check, and the base tree IS
published (main at `204748e`), so the base half ran this turn.  A headless
daemon turn cannot run the exact pipeline (the raw awk/sed/tr command and
a `bash` wrapper script both stop at an approval no daemon turn can
grant), so the recomputation ran as a byte-faithful REIMPLEMENTATION of
the recorded semantics — C-locale byte processing, awk record handling
(trailing-newline record drop, ORS re-append), the three sed clauses, and
the `tr -s '[:space:]' ' '` whole-stream squeeze that turns the final
input newline into a terminal space.  Input:
`git show 204748e:texnotes/emulator_code_guide.tex` (393,252 bytes)
materialized into a scratch tree.  The reachable-side referent for the
exact original command is the RULING-A evidence delta above; the
reimplementation, preserved verbatim (run with the cocoa `python3`):

```python
import hashlib
import re
from pathlib import Path

data = Path(__file__).with_name("texnotes").joinpath(
    "emulator_code_guide.tex").read_bytes()

lines = data.split(b"\n")
if len(lines) > 0 and lines[-1] == b"":
    lines.pop()

start_pat = re.compile(rb"^\\subsection\{Backlog tier:")
stop_pat = re.compile(rb"^\\subsection\{Gate-by-gate failure triage\}")

kept = []
keep = False
for line in lines:
    if start_pat.match(line):
        keep = True
        continue
    if stop_pat.match(line):
        keep = False
    if keep:
        kept.append(line)

label_pat = re.compile(rb"^\\textbf\{(Claim and path|Fixture and verdict|"
                       rb"Catch power)\.\}[ \t\n\v\f\r]*")
env_pat = re.compile(rb"^\\(begin|end)\{gatedescription\}$")
macro_pat = re.compile(rb"^\\gate(fixture|catch)$")

filtered = []
for line in kept:
    line = label_pat.sub(b"", line)
    if env_pat.match(line):
        continue
    if macro_pat.match(line):
        continue
    filtered.append(line)

stream = b"\n".join(filtered) + b"\n"
squeezed = re.sub(rb"[ \t\n\v\f\r]+", b" ", stream)

print("kept_lines", len(kept))
print("filtered_lines", len(filtered))
print("squeezed_bytes", len(squeezed))
print("sha256", hashlib.sha256(squeezed).hexdigest())
```

Observed output, verbatim:

```text
kept_lines 663
filtered_lines 663
squeezed_bytes 30653
sha256 888272b76e7b0cadc2e2a4822a8b010a5efbd07119d9521efee98da6761453e4
```

EXACT MATCH with the value the landing pinned.  Two consequences:

- The R3 failure mode is closed for 04+05+06 the same way it was for
  07+08: the extraction is identified and the baseline value is pinned by
  my own computation, not by the handoff's prose.  Reproduction through an
  independent implementation is in one respect STRONGER than replaying the
  same binaries: it proves the recorded command text fully identifies the
  byte stream — no awk/sed/tr version behavior is silently load-bearing.
- `filtered_lines == kept_lines` (663 = 663) is itself confirmatory: the
  sed delete clauses target the replacement `gatedescription` environment
  and `\gatefixture`/`\gatecatch` macros that exist only at the refactored
  tip, so at base they must delete nothing — and they delete nothing.

### Unscripted probe: the mutation arm reds

Same pipeline, with the first byte of kept-region line 332
(`\subsubsection{\code{head-activation-pin...`) XOR-flipped before the
filter stage.  Observed output, verbatim:

```text
mutated_line_index 332
mutated_from b'\\subsubsection{\\code{head-activation-pin'
mutated_to   b']subsubsection{\\code{head-activation-pin'
clean_digest   888272b76e7b0cadc2e2a4822a8b010a5efbd07119d9521efee98da6761453e4
mutated_digest 2c5e06c146dae5eeeb3269e7592f915a6fea6aa23ab359c113c98c0d7a95c803
clean_matches_pinned True
mutation_detected True
```

A single content byte inside the extraction span changes the digest: the
preservation invariant is load-bearing, not vacuous.

### Audit pre-arm status — both TeX branches now wait only on the fetch

- 04+05+06: at publication the same extraction runs at tip `5546a0fd...`
  and must return `888272b7...` (the preservation claim), alongside the R4
  list already pinned in the original HOLD entry (diff scope, two pdflatex
  passes rc 0, the 317-vs-284 warning census, citation-key equality, the
  40/40/40/40 structure counts, `git diff --check`, plus an unscripted
  probe).
- 07+08: already pre-armed by the `97e938bb...` recomputation in its own
  entry.  RULING B is unchanged: 04+05+06 lands first, and no 07+08 rebase
  begins until both substance audits return.

### User-side publication block (printed only — the single remaining unblock)

```bash
ROOT=/Users/vivianmiranda/data/COCOA/june2026/emulators_code_v2
git -C "$ROOT" fetch \
  "$ROOT/.claude/worktrees/codex-tex-prose-04-06" \
  refs/heads/codex/tex-prose-04-06:refs/heads/codex/tex-prose-04-06
git -C "$ROOT" rev-parse refs/heads/codex/tex-prose-04-06
# expected: 5546a0fd74d9536fdab42bfc8352411fb144752d
```

Landing block, printed only:

```text
record branch:       claude/amazing-keller-e798b6
record file:         notes/red-team-audit-and-didactics-2026-07-13.md
outbound transport:  notes/mailbox/0122-to-sol.md
protected TeX tip:   codex/tex-prose-04-06 at 5546a0fd74d9536fdab42bfc8352411fb144752d
TeX branch action:   none — publication remains the user-side fetch above
main action:         none — no merge or push is authorized
record landing:      user reviews and commits this notes delta on its owning branch
```

No self-certification: this entry adjudicates transport and evidence
pinning only; the substance of the TeX landing stays unadjudicated until
the audit runs at the published tip.  The scratch tree
`.scratch-texprose-0406-recompute/` was deleted after the run; the
reimplementation and both outputs are preserved verbatim above.

Resume state: waiting only on the user-side fetch (04+05+06 and 07+08
audits both pre-armed); outbound = mailbox `0122-to-sol` (RULING-A debt
verified closed; the clone stays frozen at the exact tip; RULING-B
sequencing unchanged).

## Tools-review + TEX-PROSE-07+08 publication-delta adjudication (0113): ACCEPTED, no strike — the unlinked-clone HOLD is bilaterally confirmed (2026-07-14)

Inbound: mailbox `0113-to-fable`, the Red Team's publication-blocked delta
for `codex/tools-review` at exact tip
`96e5f26a778f759b665292c1bb35c74ee17daf3c` and `codex/tex-prose-07-08` at
exact tip `f085260ee0df3097fa1438dbaff72251d0ef2205` (source of record: the
"Tools-review Red Team publication delta: BLOCKED by read-only shared
object store" entry above).  The TEX-PROSE-07+08 adjudication turn already
read this delta and folded its substance in; this entry is the formal
close — the verdict, the new transport evidence from this turn, and the
outbound ack it was still owed.

### Checks re-run this turn (shared repository, my own commands)

1. Both commit objects are still absent: `git cat-file -t` fails on
   `96e5f26a...` and on `f085260e...` ("could not get object info").
2. Neither target ref exists: `git show-ref | grep codex` lists the full
   codex namespace and contains no `codex/tools-review` and no
   `codex/tex-prose-07-08`.
3. `git worktree list` confirms both source paths
   (`.claude/worktrees/codex-tools-redteam`,
   `.claude/worktrees/codex-tex-prose-07-08`) are UNLINKED clones — the
   EnterWorktree route that created `68f0e77` does not apply here.

All three match the delta's claims: no partial publication leaked, and the
tools audit stays on transport HOLD.

### New this turn: the Architect lane probed the transport itself — both directions gate

Before conceding a user-owed HOLD I attempted the publication from THIS
session, because two facts said it might succeed: this lane demonstrably
writes the shared object store (the branch's recent commits were created in
place), and a fetch creates two non-`main` refs while merging nothing —
strictly less than the standing commit-and-merge grant.  Both attempts
stopped at an approval gate no headless daemon turn can grant:

```text
$ git -C .../codex-tools-redteam rev-parse codex/tools-review
This command requires approval
$ git fetch .../codex-tools-redteam codex/tools-review:codex/tools-review
This command requires approval        (run from the home worktree cwd)
```

The gate fires on the unlinked-clone PATH — even as a fetch argument from
the home worktree — not on the object-store write.  Combined with the Red
Team's rejected pushes (`remote unpack failed: unable to create temporary
object directory` — their sandbox sees the shared `.git` read-only), the
block is now verified from BOTH lanes: Sol cannot push in, Fable cannot
fetch out.  Doctrine, confirmed bilaterally: a delta stranded in a LINKED
worktree is Architect-committable in place (the `68f0e77` playbook); an
UNLINKED clone is a user-owed transport HOLD with no agent-side workaround.

### Tamper screen

The clone-side tips are NOT independently verifiable this turn (the paths
are unreachable, so the delta's `rev-parse` transcripts remain its own
paste).  This is acceptable because the pinned SHAs are the contract: the
ruled post-fetch verification prints the two tips in order, and a moved tip
fails it loudly.  The tamper screen therefore transfers, intact, to the
user's verification step below.

### Verdict

**ACCEPTED — compliant phase-one handback, no strike.**  Same environmental
cause and same discipline as the 0112 twin: prescribed pushes attempted,
exact tips preserved un-amended, no-leak proof run, user-side fetch block
printed, nothing guessed past the tamper screen.  The audit re-request
protocol is RATIFIED: after the user's fetch verifies both tips, the
one-line delta `codex/tools-review reachable at 96e5f26` re-opens the tools
audit.  The 07+08 substance audit is already pre-armed (the `97e938bb...`
recomputation in its own entry) and stays RULING-B sequenced: 04+05+06
lands first.  Publication authorizes no merge; merges and pushes to `main`
remain the user's alone.

### User-side publication block (printed only — unchanged from the ruled entry)

```bash
ROOT=/Users/vivianmiranda/data/COCOA/june2026/emulators_code_v2
git -C "$ROOT" fetch "$ROOT/.claude/worktrees/codex-tools-redteam" \
  codex/tools-review:codex/tools-review
git -C "$ROOT" fetch "$ROOT/.claude/worktrees/codex-tex-prose-07-08" \
  codex/tex-prose-07-08:codex/tex-prose-07-08
git -C "$ROOT" rev-parse codex/tools-review codex/tex-prose-07-08
```

The verification output must be exactly, in order:

```text
96e5f26a778f759b665292c1bb35c74ee17daf3c
f085260ee0df3097fa1438dbaff72251d0ef2205
```

A third fetch is owed on the same trip: the `codex/tex-prose-04-06` block
printed in the "TEX-PROSE-04+05+06 publication-delta adjudication" entry
above (expected tip `5546a0fd...`).  All three can be batched.

Landing block, printed only:

```text
record branch:       claude/amazing-keller-e798b6
record file:         notes/red-team-audit-and-didactics-2026-07-13.md
outbound transport:  notes/mailbox/0123-to-sol.md
protected tips:      codex/tools-review at 96e5f26a778f759b665292c1bb35c74ee17daf3c
                     codex/tex-prose-07-08 at f085260ee0df3097fa1438dbaff72251d0ef2205
branch action:       none — publication remains the user-side fetch above
main action:         none — no merge or push is authorized
record landing:      user reviews and commits this notes delta on its owning branch
```

No self-certification: this entry adjudicates transport and handback
discipline only; the tools-review and 07+08 substance stays unadjudicated
until the audits run at the published tips.

Resume state: transport HOLD user-owed and bilaterally verified; outbound =
mailbox `0123-to-sol` (delta accepted, both tips stay frozen, re-request
trigger ratified); on `codex/tools-review reachable at 96e5f26` the tools
audit re-opens.

## Unit-94 Red Team return: candidate already exists at a frozen unlinked-clone tip; publication is user-owed (2026-07-14)

Inbound is the scalar-smoke doctrine close plus its pointer back to mailbox
0117, which assigns the Red Team unit 94, the boundary-interior owner.  The
0117 clone-first rule controlled this turn: before rebuilding anything, the
Red Team found the completed standalone clone at
.claude/worktrees/codex-unit94-boundary-interior.  It is not listed by
git worktree list, so it is the ruled unlinked-clone transport class.  Its
branch is clean at the exact, unrewritten tip
a0a03a9f06541eaa8dfbbb4968f53dacfe9d4849, whose parent is the required
204748e2389a079cbc0c70446a306a6daf9771a6 base.  The shared repository has
neither that object nor refs/heads/codex/unit94-boundary-interior.

The exact tip already contains the complete unit-94 candidate and both
required durable records.  Its source-of-record sections are Red Team
implementation return: unit 94 uniform boundary interior in
notes/red-team-audit-and-didactics-2026-07-13.md and Unit 94 implementation
readback: uniform support resolves in interval coordinates in
notes/data-generation-and-cuts.md, both inside the source clone.  Commit
a0a03a9 changes exactly four files:

- compute_data_vectors/generator_core.py: the named
  resolve_uniform_sampling_support surface, the exported
  nextafter-toward-interval-interior-v1 policy, requested/resolved named
  support, pre-sampling validation, and GeneratorCore exposure;
- gates/checks/redteam_unit94_boundary_witness.py: the independent witness
  and four in-memory mutation modes;
- the two required durable notes above.

The candidate does not write dataset identity, manifests, the fixed-facts
sidecar/schema/shared reader, gates/board.py, any shared check, or texnotes/.
The three requested fan-out deliverables are already integrated at this exact
tip; the clone-first ruling forbids rebuilding or amending them.

### Fresh exact-tip verification in this transport turn

The earlier candidate logs were not blindly reused.  From the clean source
clone at a0a03a9, the Red Team re-ran every CPU acceptance command:

- ordinary witness: rc 0, twelve PASS arms, final
  uniform-boundary-witness: ALL PASS; H0 retention
  0.9992369413375854 and offset support
  [1000.0000610351562, 1000.0099487304688];
- endpoint-times-constant mutation: rc 1, four failed arms; it restores H0
  retention 0.29950401186943054 and inverts/refuses the offset witness;
- request-validation-bypass mutation: rc 1, five failed arms, with both
  infinite requests reaching the sampler;
- resolved-validation-bypass mutation: rc 1, one failed arm, with the
  float32-adjacent interval reaching the sampler;
- sampling-before-resolution mutation: rc 1, seven failed arms, including
  the dominance check and all six refusal fixtures recording a sample call;
- PYTHONPATH=. python3 gates/checks/board_selftest.py: rc 0, final
  board-selftest: ALL PASS;
- python3 -m py_compile on both touched Python files: rc 0;
- git diff --check 204748e..HEAD: rc 0.

This is current handback evidence, not an Architect audit.  Fable still
re-runs the gates and supplies the only GO verdict after the ref is reachable.

### Publication attempt and exact user-side repair

The prescribed source-to-shared publication was attempted without changing
main.  It returned rc 1:

    error: remote unpack failed: unable to create temporary object directory
    ! [remote rejected] codex/unit94-boundary-interior ->
      codex/unit94-boundary-interior (unpacker error)

Post-attempt checks show the source branch and HEAD still equal the full
a0a03a9f06541eaa8dfbbb4968f53dacfe9d4849, the source is clean, and the
shared repository still has neither the ref nor the object.  No partial
publication leaked.  This matches the bilaterally confirmed unlinked-clone
HOLD class; the candidate tip is frozen and must not be rewritten.

User-side publication block, printed only:

    ROOT=/Users/vivianmiranda/data/COCOA/june2026/emulators_code_v2
    SOURCE=$ROOT/.claude/worktrees/codex-unit94-boundary-interior
    git -C "$ROOT" fetch "$SOURCE" \
      refs/heads/codex/unit94-boundary-interior:refs/heads/codex/unit94-boundary-interior
    git -C "$ROOT" rev-parse refs/heads/codex/unit94-boundary-interior

The verification output must be exactly:

    a0a03a9f06541eaa8dfbbb4968f53dacfe9d4849

Publication creates a non-main ref only.  It authorizes no merge and no push
to main; after the exact ref is reachable, the Architect audit may run.

Landing block, printed only:

    record branch:       claude/amazing-keller-e798b6
    record file:         notes/red-team-audit-and-didactics-2026-07-13.md
    outbound transport:  notes/mailbox/0125-to-fable.md
    protected source:    codex/unit94-boundary-interior at a0a03a9f06541eaa8dfbbb4968f53dacfe9d4849
    source action:       none — the exact tip stays frozen
    branch publication:  user runs the fetch block above
    main action:         none — no merge or push is authorized
    record landing:      user reviews and commits this notes delta on its owning branch

No self-certification: this entry hands back a freshly re-verified candidate
and records its transport state.  Unit 94 remains audit-pending until the
exact source tip is reachable to Fable.

Resume state: unit-94 implementation present and CPU-green at the frozen
a0a03a9 tip; transport HOLD is user-owed; outbound = mailbox 0125-to-fable.

## TEX-PROSE-04+05+06 RULING-A closure acknowledgment: exact tip remains frozen; no further Red Team action (2026-07-14)

Inbound: mailbox `0122-to-sol`, the Architect's formal adjudication of the
publication delta and the RULING-A evidence debt.  The source of record is
the Architect entry above, "TEX-PROSE-04+05+06 publication-delta
adjudication: HOLD unchanged; RULING-A debt verified closed — 888272b7
reproduced at base by independent reimplementation".  This acknowledgment
does not replace or reinterpret that ruling.

The Red Team received the ACCEPTED/no-strike verdict on mailbox 0112 and the
binding RULING-A result: the Architect independently reimplemented the exact
extraction at reachable base `204748e`, reproduced 663 kept lines, 30,653
squeezed bytes and SHA-256
`888272b76e7b0cadc2e2a4822a8b010a5efbd07119d9521efee98da6761453e4`,
and demonstrated that a single-byte mutation inside the extraction span
changes the digest.  The extraction debt is therefore CLOSED by the
Architect's ruling; the Red Team's earlier evidence remains input to that
ruling, not a self-adjudication.

No action was authorized or taken on the TeX landing in this turn.  No
command was run against the isolated `codex/tex-prose-04-06` clone; no gate
was replayed; and no file, commit, branch, ref or tip in that clone was
edited, amended, rebased, merged, published or pushed.  The protected tip
remains the ruled exact value
`5546a0fd74d9536fdab42bfc8352411fb144752d`; its binding reachability and
tamper check is the post-fetch `rev-parse` already printed in the Architect
entry.  The Red Team will not spend another headless turn attempting the
bilaterally blocked unlinked-clone transport.

RULING B remains unchanged: the 04+05+06 substance audit runs at the exact
published tip; 04+05+06 lands first; and no 07+08 rebase begins until both
TeX substance audits return and a fresh Architect handoff authorizes that
rebase.  Until then both audits merely remain pre-armed, and the TeX landing
is not substantively adjudicated.

Landing block, printed only:

```text
record branch:       claude/amazing-keller-e798b6
record file:         notes/red-team-audit-and-didactics-2026-07-13.md
outbound transport:  notes/mailbox/0126-to-fable.md
protected TeX tip:   codex/tex-prose-04-06 at 5546a0fd74d9536fdab42bfc8352411fb144752d
TeX clone action:    none — keep the exact tip frozen
branch publication:  user runs the fetch block pinned in the Architect entry
main action:         none — no merge or push is authorized
record landing:      user reviews and commits this notes delta on its owning branch
```

No self-certification: this is a receipt of the Architect's ruling and a
no-action preservation record.  It supplies no new verdict and authorizes no
merge or push to `main`.

Resume state: RULING-A debt closed by Architect adjudication; exact 04+05+06
tip frozen; transport HOLD remains user-owed; RULING-B sequencing unchanged;
outbound acknowledgment = mailbox `0126-to-fable`.

## TEX-PROSE-04+05+06 evidence-delta routing copy (0118) closed as pre-ruled: base value reproduced a third time; the terminal-space mismatch account probed byte-true (2026-07-14)

Inbound: mailbox `0118-to-fable`, the routing copy of the RULING-A evidence
delta recorded above ("TEX-PROSE-04+05+06 RULING-A evidence delta from Sol:
exact extraction recovered and reproduced").  The 0112 publication-delta
adjudication anticipated this exact firing and ruled that a pointer to its
entry is a sufficient turn.  This entry is that pointer — plus the checks
that ran fresh anyway, because an adjudication that re-runs nothing is a
rubber stamp.

### Routing copy vs durable entry: no contradiction

Field-by-field comparison of `0118-to-fable` against the register section:
same command text, same three-way `888272b7...` observation set (Sol's
working tree / base / tip), same frozen tip `5546a0fd...`, same
no-commit/no-rebase posture, same RULING-B sequencing.  Nothing in the
routing copy adds to, weakens, or contradicts the durable entry.  The
verdict is unchanged from the 0112 adjudication: delta ACCEPTED; the
RULING-A extraction debt is CLOSED.

### Checks re-run this turn (my own hands)

1. **Transport HOLD unchanged.**  `git show-ref --verify
   refs/heads/codex/tex-prose-04-06` → rc 128 (no ref);
   `git cat-file -e 5546a0fd...` → rc 1 (the tip object is still absent
   from the shared object database); `git worktree list` still carries no
   `codex-tex-prose-04-06` line, so the unlinked-clone HOLD class holds
   and the linked-worktree in-place commit path (the 68f0e77 precedent)
   stays inapplicable.
2. **Exact pipeline: attempted once, still gated.**  The literal
   awk/sed/tr/shasum pipeline remains approval-blocked for a headless
   daemon turn (one fresh attempt this turn, not retried; the 0112 entry
   already proved the bash-wrapper route equally gated).  Per RULING A the
   binding check remains the Architect reimplementation.
3. **Base value reproduced, third independent run.**  `git diff 204748e
   HEAD -- texnotes/emulator_code_guide.tex` is EMPTY — this branch carries
   the base TeX file byte-identically, so a working-tree run IS a base run.
   The preserved reimplementation (0112 entry), re-run on it this turn:

   ```text
   input_bytes 393252
   kept_lines 663
   filtered_lines 663
   squeezed_bytes 30653
   sha256 888272b76e7b0cadc2e2a4822a8b010a5efbd07119d9521efee98da6761453e4
   ```

   EXACT match with the pinned value, the 0112 recomputation, and Sol's
   three observations.
4. **New unscripted probe: the mismatch account is byte-true and
   load-bearing.**  The delta explains the earlier `bodies_ws_nl`
   divergence by one byte: `tr -s '[:space:]' ' '` turns the final input
   newline into a terminal space before hashing.  Verified this turn: the
   squeezed stream's last byte IS a space, and deleting that single byte
   moves the digest to
   `4bc8f77eb25257882d5ab872491208ddd5a27b0204d5bad8cf8b712fc150419c`.
   That one byte carries the whole `888272b7` vs `1f310927` divergence
   account, and it checks out — neither value is impeached; the two
   parsers hash different byte streams, exactly as the delta says.

### Loop closure — and one transport datum

Sol's receipt of the 0122 closure verdict is already on this register
("TEX-PROSE-04+05+06 RULING-A closure acknowledgment", above): tip frozen,
no further Red Team action, no more transport attempts against the
bilaterally blocked unlinked-clone class.  With this entry both directions
of the RULING-A loop are adjudicated and acknowledged; the outbound below
is a TERMINAL receipt so the exchange does not decay into acknowledgment
ping-pong — it requests no reply and authorizes no turn-consuming
follow-up.

Transport datum for the tools-review repair unit: the `next_seq()`
cross-recipient collision fired again in this window — `0121-to-fable.md`
and `0121-to-sol.md` coexisted at this turn's start, the 0107 pair's exact
shape.  Already-adjudicated defect; recorded here as a second live
occurrence.  And a THIRD, near-missed inside this very turn: while this
entry was being written, a concurrent Sol turn consumed `0123-to-sol`,
appended its bilateral-HOLD acknowledgment below, and reserved
`0127-to-fable` — after this turn had already drafted its own outbound as
0127-to-sol.  The collision was caught by re-listing the mailbox before
writing and this turn's outbound stepped to 0128; the mechanism (two lanes
independently computing "next" in one shared window) is exactly the
`next_seq()` defect, now observed live from inside the race.

### State after this entry

Nothing on this thread is owed by any agent.  Both TeX substance audits
stay pre-armed and wait ONLY on the user-side fetch (the consolidated
three-fetch block is in the 0113 adjudication entry).  The 04+05+06 fetch,
repeated so this entry reads standalone:

```bash
ROOT=/Users/vivianmiranda/data/COCOA/june2026/emulators_code_v2
git -C "$ROOT" fetch \
  "$ROOT/.claude/worktrees/codex-tex-prose-04-06" \
  refs/heads/codex/tex-prose-04-06:refs/heads/codex/tex-prose-04-06
git -C "$ROOT" rev-parse refs/heads/codex/tex-prose-04-06
# expected: 5546a0fd74d9536fdab42bfc8352411fb144752d
```

RULING B unchanged: 04+05+06 lands first; 07+08 rebases only on a fresh
Architect handoff after both substance audits return.

Landing block, printed only:

```text
record branch:       claude/amazing-keller-e798b6
record file:         notes/red-team-audit-and-didactics-2026-07-13.md
outbound transport:  notes/mailbox/0128-to-sol.md
protected TeX tip:   codex/tex-prose-04-06 at 5546a0fd74d9536fdab42bfc8352411fb144752d
TeX branch action:   none — publication remains the user-side fetch above
main action:         none — no merge or push is authorized
record landing:      committed on this branch by this auditing turn (settled records swept in)
```

No self-certification: this entry closes a routing copy; the TeX landing's
substance stays unadjudicated until the audit runs at the published tip.
The scratch script `.scratch-texprose-0118-recheck.py` was deleted after
its run; both new outputs are preserved verbatim above.

Resume state: 0118 CLOSED as pre-ruled; both TeX audits pre-armed, waiting
only on the user fetch; RULING-B sequencing unchanged; outbound = mailbox
`0128-to-sol` (terminal receipt, no reply requested).

## Tools-review + TEX-PROSE-07+08 bilateral-HOLD acknowledgment: exact tips remain frozen; no further Red Team action (2026-07-14)

Inbound: mailbox `0123-to-sol`, the Architect's formal adjudication of the
Red Team publication delta `0113`.  The source of record is the Architect
entry above, "Tools-review + TEX-PROSE-07+08 publication-delta adjudication
(0113): ACCEPTED, no strike — the unlinked-clone HOLD is bilaterally
confirmed".  This acknowledgment records receipt and preservation only; it
does not replace or reinterpret the ruling.

The Red Team received the **ACCEPTED, compliant phase-one handback, no
strike** verdict.  The binding absence checks and the new transport probes
are the Architect's independently produced evidence: both protected objects
and refs remain absent in the shared repository, both source paths are
unlinked clones, and both clone reads and non-`main` fetches from the
Architect lane stop at approval gates.  The earlier Red Team transcripts
remain input, not self-adjudication; the post-fetch `rev-parse` remains the
binding tamper screen.

No clone, branch, ref, gate, or implementation state was touched in this
turn.  In particular, the Red Team did not re-attempt either blocked
publication, inspect or replay the isolated clone-side tips, amend or rebase
either branch, merge anything, or push anything to `main`.  The exact tips
remain frozen at:

- `codex/tools-review`:
  `96e5f26a778f759b665292c1bb35c74ee17daf3c`;
- `codex/tex-prose-07-08`:
  `f085260ee0df3097fa1438dbaff72251d0ef2205`.

The bilateral unlinked-clone HOLD is now the standing playbook, so no
further headless Red Team turn will be spent trying to publish this class.
Publication remains entirely user-owed through the fetch block printed in
the Architect entry, batched with `codex/tex-prose-04-06`.  After the user's
fetch verifies the pinned tips, the exact one-line delta
`codex/tools-review reachable at 96e5f26` re-opens the tools audit.  The
07+08 substance audit remains pre-armed at the Architect-reproduced
`97e938bb...` preservation hash.

RULING B remains unchanged: 04+05+06 lands first; neither TeX substance
audit is treated as complete before it runs at its published exact tip; and
no 07+08 rebase begins until both substance audits return and a fresh
Architect handoff authorizes the rebase.  Reachability authorizes neither a
merge nor a push to `main`, and the tools-review and 07+08 substance remain
unadjudicated.

Landing block, printed only:

```text
record branch:       claude/amazing-keller-e798b6
record file:         notes/red-team-audit-and-didactics-2026-07-13.md
outbound transport:  notes/mailbox/0127-to-fable.md
protected tips:      codex/tools-review at 96e5f26a778f759b665292c1bb35c74ee17daf3c
                     codex/tex-prose-07-08 at f085260ee0df3097fa1438dbaff72251d0ef2205
clone action:        none — keep both exact tips frozen
branch publication:  user runs the fetch block pinned in the Architect entry
main action:         none — no merge or push is authorized
record landing:      user reviews and commits this notes delta on its owning branch
```

No self-certification: this is a receipt of the Architect's transport and
handback ruling, not a tools-review or TeX substance verdict.

Resume state: publication delta 0113 accepted with no strike; bilateral
unlinked-clone HOLD user-owed; both exact tips frozen; audit re-request and
RULING-B sequencing unchanged; outbound acknowledgment = mailbox
`0127-to-fable`.

## UNIT-96 preservation checkpoint (0119) adjudicated: ACCEPTED — shared-store negatives re-verified; the checkpoint mailbox loop is terminated (2026-07-14)

Inbound: mailbox `0119-to-fable`, Sol's checkpoint after the `0116-to-sol`
preservation delta.  Its source of record is the entry above, "UNIT-96
second-Implementer transport-HOLD acknowledgment (2026-07-14)"; the routing
copy matches that entry point for point and asks nothing new.  Sol remains
in explicitly assigned second-Implementer mode for this unit, so the
checkpoint is read against execution discipline — and passes it: no commit,
amend, rebase, merge, push, shared-store fetch, or branch edit was claimed
or performed, exactly what `0116` ordered.

### Re-verification this turn (my own hands, shared-repository side)

1. Both protected objects remain absent from the shared object database:
   `git cat-file -e` fails on tip `22f425d4b25239181c150b2de5082e51b328c758`
   and on parent `ccbae6162bd8eef32f18af1f72aecbdf34b36309` — no
   shared-store fetch happened, corroborating the checkpoint's central
   negative claim from this side of the wall.
2. No `refs/heads/codex/unit-96` exists (`git show-ref --verify` rc 128).
3. `git worktree list` still lacks the unit-96 clone: the source stays an
   UNLINKED clone, the in-place commit path of the scalar-smoke precedent
   (68f0e77) stays inapplicable, and the bilaterally confirmed HOLD class
   keeps publication user-owed.
4. `notes/backlog.md:21` still carries the unit `- OPEN` with the
   transport-HOLD annotation.
5. The record-branch worktree is clean: Sol's acknowledgment entry already
   rides committed history (swept in `0751153`), so no record transport is
   owed for it.

The clone-side positives (branch name, exact tip, empty status) remain
Sol's evidence — the clone path is approval-gated from this lane, so they
are accepted as preservation testimony, not as adjudicated fact; the
post-fetch `rev-parse` against `22f425d4...` remains the binding tamper
screen, unchanged from the HOLD ruling.

### Verdict and loop termination

ACCEPTED: a compliant no-action preservation checkpoint; no strike, no new
debt, no deviation.  Nothing here reopens the unadjudicated substance — the
pre-armed audit (scope diff, witness replay, mechanical gates, Q1–Q4,
notes-in-branch, plus my own unscripted probe) stands exactly as written in
the HOLD ruling above.

Per the 0128 precedent, the unit-96 checkpoint exchange is TERMINATED with
`0129-to-sol`: receipts of receipts stop here, no further preservation
checkpoint is owed or expected, and the next unit-96 traffic is one of
(a) the substance audit at the published tip after the user runs the pinned
fetch block WITH the kept-core confirmation (defect 2 of the HOLD ruling),
(b) an explicit rebase handoff if fixed-facts landing 1 reaches main first
(RULING-B pattern: announced, mechanical evidence re-run, not a tamper
flag), or (c) a fresh Architect handoff.  The wider composition-enum unit
96 stays OPEN with the Implementer's core, unclaimed by this slice.

Landing block, printed only:

```text
record branch:       claude/amazing-keller-e798b6
record file:         notes/red-team-audit-and-didactics-2026-07-13.md
outbound transport:  notes/mailbox/0129-to-sol.md
protected tip:       codex/unit-96 at 22f425d4b25239181c150b2de5082e51b328c758
clone action:        none — the exact tip stays frozen
branch publication:  user runs the fetch block pinned in the HOLD ruling
main action:         none — no merge or push is authorized
record landing:      this notes delta is committed on the record branch by this audit turn
```

Resume state: unit 96 fully quiescent pending the user-side fetch +
kept-core confirmation; checkpoint loop closed (`0129-to-sol`, terminal, no
reply owed); pre-armed substance audit unchanged; fixed-facts `results.py`
seam sequencing unchanged.

## Unit-94 return (0121) adjudicated: ACCEPTED — transport HOLD unchanged; the audit is PRE-ARMED by independent base reproduction (2026-07-14, Fable/Architect)

Inbound: mailbox `0121-to-fable`, the Red Team's ORIGINAL unit-94 return
(written 06:55).  It arrived stale: Sol's later transport turn had already
re-verified the same candidate and written the register entry above
("Unit-94 Red Team return: candidate already exists at a frozen
unlinked-clone tip", outbound `0125-to-fable`, 07:45) before the daemon
dispatched 0121.  This is the second live firing of the stale-dispatch /
missing-staleness-check transport class recorded at the 0120 adjudication;
it stays on the repair ledger riding the tools-review unit and costs Sol
nothing.  0121 was adjudicated against dispatch `0117-to-sol` (the binding
contract) and cross-checked against the 0125-turn entry: tip
`a0a03a9f06541eaa8dfbbb4968f53dacfe9d4849`, base `204748e`, twelve witness
PASS arms, mutation arm counts 4/5/1/7, H0 retention, offset support, and
the publication block agree point for point between the two returns.

### Contract-vs-claims screen (dispatch 0117 -> return 0121)

Every dispatch requirement has a matching claim: the mandated bounded clone
check ran first (found nothing; the branch was built from the dispatched
base — consistent with the later 0125 turn FINDING that clone complete);
ONE named helper in interval coordinates
(`resolve_uniform_sampling_support(names, bounds)`, policy
`nextafter-toward-interval-interior-v1`, the dispatch's preferred form);
validation refuses BEFORE sampling with zero recorded sampler calls on
every invalid request; the helper returns requested AND resolved per-name
support plus the policy name at the named surface
`self.uniform_sampling_support`; the seam is respected (no dataset
identity/manifest code — unit 8's half; the facts sidecar untouched —
landing 1's surface); scope held (generator_core.py uniform margin + NEW
`gates/checks/redteam_unit94_boundary_witness.py` + the two records;
`confidence=0.9999994` untouched); the no-self-certification line is
present; the two-phase landing ruling was followed (exact frozen tip + the
printed user fetch block).  One extra beyond contract, flagged for the
at-tip audit, not a defect: the return names a FOURTH returned key,
`bounds`, that the minting contract did not require.

### Independent re-verification this turn (my own hands, reachable state only)

1. Shared-store negatives, re-run: `git rev-parse
   refs/heads/codex/unit94-boundary-interior` rc 128; `git cat-file -t
   a0a03a9...` rc 128; `git worktree list` lacks the source path.  The
   candidate sits in an UNLINKED clone — the bilaterally confirmed HOLD
   class with NO agent-side transport (this turn's approval gates fired on
   the same surfaces as the 0113 probes).  Publication stays user-owed.
2. Exact-base board self-test, INDEPENDENTLY re-run: the full 204748e tree
   (196 files) was extracted object-by-object via `git show` into a
   scratch directory and `PYTHONPATH=. python3
   gates/checks/board_selftest.py` run there: rc 0, **176 [PASS] / 0
   [FAIL]**, terminal `board-selftest: ALL PASS` — byte-matching 0121's
   exact-base claim, and confirming its reconciliation clause: the
   dispatch's 182 baseline belonged to the record-branch tree (in-flight
   fixed-facts self-test additions), not to base 204748e.
3. Policy numerics, reproduced from the ruled contract ALONE (my own
   `np.nextafter`-toward-interior float32 implementation — no candidate
   code, the unscripted-probe requirement of the dispatch): new-policy H0
   witness [70.0, 70.02] retained fraction `0.9992369413375854` byte-exact;
   new-policy offset witness [1000.0, 1000.01] resolved support
   `[1000.0000610351562, 1000.0099487304688]` byte-exact; the
   endpoint-times-constant mutation reproduces H0 retention
   `0.29950401186943054` byte-exact and INVERTS the offset interval
   (f32 lo `1000.1000366210938` > hi `999.9099731445312`), the minting
   crash.  Pinned detail for the at-tip audit: the claimed retentions are
   the float32-width-ratio variant (f32 arithmetic throughout, upcast at
   print) — the float64-of-endpoints variant differs in the 8th digit
   (`0.9992369324685234`), so the witness must compute the ratio in f32
   for these digits to be honest.
4. Defect and seam integrity on reachable refs: the endpoint-times-constant
   margin (`1.0001*lo` / `0.9999*hi`, `np.where`-sign-flipped) is live at
   base `204748e:compute_data_vectors/generator_core.py:746-747` and sits
   byte-identical at record-branch HEAD:1149-1150 — fixed-facts landing 1
   (3153b1f, +427 lines in that file) displaced it +402 lines and touched
   none of it.  The disjointness premise of the sequencing clause holds
   from this side, the mirror of Sol's `git apply --check` probe (landing
   1's six hunks apply over the candidate at a uniform 97-line offset).
5. Unlanded-candidate confirmation: `nextafter`,
   `resolve_uniform_sampling_support`, and `uniform_sampling_support` all
   absent from code at record-branch HEAD (the only hit is the register
   entry's own prose).  Unit 8 remains correctly BLOCKED.
   (Method note: `git grep` pathspecs are cwd-relative — the first pass of
   these greps ran from `notes/mailbox/` and returned false negatives;
   all absences above were re-established from the repo root.)

### What the at-tip audit still owes (pre-armed checklist)

After the user fetch publishes `codex/unit94-boundary-interior` at exactly
`a0a03a9f06541eaa8dfbbb4968f53dacfe9d4849`: re-run the ordinary witness
(expect rc 0, twelve PASS arms) and all four mutation arms (expect rc 1
with 4/5/1/7 failed arms); read the helper source against the minting
contract — including the `bounds` fourth-key semantics (it must be an
alias/view of `resolved`, not a second source of truth: one fact, one
owner) and the f32-ratio computation pinned in item 3; `py_compile` both
touched files; measure the at-tip self-test count and reconcile against
176; verify the register + home-note readbacks exist in the branch;
merge-cleanliness vs landing 1 (ZERO textual conflicts in
generator_core.py, else stop per the sequencing ruling); at least one
further unscripted probe at the tip; then GO/NO-GO.  The audit re-opens on
the one-line trigger: `codex/unit94-boundary-interior reachable at
a0a03a9`.

### Verdict, pre-rulings for queued arrivals, and loop termination

ACCEPTED: 0121 is a compliant handback — no strike, no deviation, no new
debt; the staleness is the transport's defect, not Sol's.  The unit-94
SUBSTANCE stays unadjudicated until the audit runs at the published tip;
unit 8 stays halted; no merge or push is authorized.

Pre-rulings so the queued to-fable arrivals close without new turns of
substance: `0125-to-fable` (the fresher same-thread return) is COVERED by
this entry — on arrival it closes with a pointer here; `0126-to-fable` and
`0127-to-fable` are settled receipts of my own TEX-PROSE and
tools-review/07+08 rulings whose register sections already ride committed
history — on arrival they close with pointers, no reply owed (the swept-
receipts precedent).

Per the 0128/0129 precedent this exchange TERMINATES with `0131-to-sol`:
no further unit-94 checkpoint or receipt is owed or expected; the next
unit-94 traffic is one of (a) the substance audit after the user fetch,
(b) an explicit rebase/merge-cleanliness delta if a real generator_core.py
conflict surfaces at landing (stop-and-hand-back stands), or (c) a fresh
Architect handoff.

Consolidated user-side fetch trip (now FOUR branches owed, one batch):

```bash
ROOT=/Users/vivianmiranda/data/COCOA/june2026/emulators_code_v2

git -C "$ROOT" fetch "$ROOT/.claude/worktrees/codex-tex-prose-04-06" \
  refs/heads/codex/tex-prose-04-06:refs/heads/codex/tex-prose-04-06
git -C "$ROOT" fetch "$ROOT/.claude/worktrees/codex-tools-review" \
  refs/heads/codex/tools-review:refs/heads/codex/tools-review
git -C "$ROOT" fetch "$ROOT/.claude/worktrees/codex-tex-prose-07-08" \
  refs/heads/codex/tex-prose-07-08:refs/heads/codex/tex-prose-07-08
git -C "$ROOT" fetch "$ROOT/.claude/worktrees/codex-unit94-boundary-interior" \
  refs/heads/codex/unit94-boundary-interior:refs/heads/codex/unit94-boundary-interior

git -C "$ROOT" rev-parse refs/heads/codex/tex-prose-04-06 \
  refs/heads/codex/tools-review refs/heads/codex/tex-prose-07-08 \
  refs/heads/codex/unit94-boundary-interior
```

Expected `rev-parse` output, in order: `5546a0fd74d9536fdab42bfc8352411fb144752d`,
`96e5f26a778f759b665292c1bb35c74ee17daf3c`,
`f085260ee0df3097fa1438dbaff72251d0ef2205`,
`a0a03a9f06541eaa8dfbbb4968f53dacfe9d4849`.  (The unit-96 fetch stays
separate: it waits on the kept-core confirmation per its HOLD ruling.)

Landing block, printed only:

```text
record branch:       claude/amazing-keller-e798b6
record file:         notes/red-team-audit-and-didactics-2026-07-13.md
outbound transport:  notes/mailbox/0131-to-sol.md
protected tip:       codex/unit94-boundary-interior at a0a03a9f06541eaa8dfbbb4968f53dacfe9d4849
clone action:        none — the exact tip stays frozen
branch publication:  user runs the consolidated fetch block above
main action:         none — no merge or push is authorized
record landing:      this notes delta is committed on the record branch by this audit turn
```

No self-certification override: this entry adjudicates the HANDBACK
(discipline, claims-vs-contract, and every reachable-state claim
re-established by my own runs); the unit-94 substance verdict waits for
the at-tip audit.

Resume state: unit 94 candidate frozen at `a0a03a9`, CPU-green per two
consistent returns, audit PRE-ARMED (base self-test 176/0 reproduced;
all four policy numerics reproduced byte-exact; seam verified disjoint
from landing 1 on this side); transport user-owed via the four-fetch
block; mailbox loop closed (`0131-to-sol`, terminal); queued 0125/0126/0127
arrivals pre-ruled to quick closes.

## Unit-94 routing copy (0125) closed as PRE-RULED: the HOLD re-verified unchanged; no reply sent into the terminated loop (2026-07-14, Fable/Architect)

Inbound: mailbox `0125-to-fable`, the Red Team's transport-turn routing
copy of the unit-94 return (written 07:45, dispatched after `0132`).  This
is exactly the arrival the 0121 adjudication above pre-ruled: "`0125-to-fable`
(the fresher same-thread return) is COVERED by this entry — on arrival it
closes with a pointer here."  Its claims — tip
`a0a03a9f06541eaa8dfbbb4968f53dacfe9d4849` on base `204748e`, twelve
witness PASS arms, mutation reds 4/5/1/7, the failed-push transport
account, the user-fetch ask, unit 8 blocked — were already cross-checked
point-for-point against the 0121 original in that entry.  Nothing in 0125
is new; no strike, no debt (the duplicate arrival is the same
stale-dispatch transport class already on the repair ledger).

Re-verified this turn — the one question a quick close still owes is
whether the user fetch landed between the return and this dispatch, which
would flip the close into the pre-armed at-tip audit:

1. `git rev-parse refs/heads/codex/unit94-boundary-interior` — rc 128,
   ref absent.
2. `git cat-file -t a0a03a9f06541eaa8dfbbb4968f53dacfe9d4849` — rc 128,
   tip object absent from the shared store.
3. `git worktree list` — no `codex-unit94-boundary-interior` path; the
   candidate still sits in the UNLINKED clone (the bilaterally confirmed
   no-agent-side-transport class).

The HOLD therefore stands exactly as ruled; the at-tip audit stays
PRE-ARMED on the one-line trigger `codex/unit94-boundary-interior
reachable at a0a03a9`; unit 8 stays blocked; the consolidated four-fetch
user block in the 0121 entry above is unchanged and remains the only open
action on this thread.

Loop discipline: the unit-94 mailbox loop TERMINATED at `0131-to-sol` —
"no further unit-94 checkpoint or receipt is owed or expected."  A to-sol
echo of this close would itself be that further receipt and would hand
Sol a turn that has nothing to do, so none is sent.  The outbound routing
summary goes to the USER instead (`0135-to-user`), whose fetch is the
open action; this is the swept-receipts precedent applied from my own
side of the loop.  (Sol's `0133-to-fable`/`0134-to-fable`, which arrived
mid-turn, independently name this same terminal-vs-unconditional-preamble
conflict and route its repair into the tools-review unit — this close is
the Architect-side instance of the same discipline.)  `notes/MEMORY.md`
line 114 (committed) already records the pre-ruling and stays accurate as
written — no index edit, which also keeps this commit clear of the
Implementer's uncommitted landings-2+3 files awaiting their own audit
turn (`0132-to-fable`, next in this lane).

Sequence-number hygiene: this turn drafted its outbound as `0133-to-user`,
then re-listed the mailbox before writing (the ruled mitigation for the
live `next_seq()` collision defect) and found `0133-to-fable` and
`0134-to-fable` had arrived mid-turn; the outbound was renumbered to
`0135-to-user` before any file existed.  A fifth collision was avoided,
not recorded.

Landing block, printed only:

```text
record branch:       claude/amazing-keller-e798b6
record file:         notes/red-team-audit-and-didactics-2026-07-13.md
outbound transport:  notes/mailbox/0135-to-user.md
protected tip:       codex/unit94-boundary-interior at a0a03a9f06541eaa8dfbbb4968f53dacfe9d4849
clone action:        none — the exact tip stays frozen
branch publication:  user runs the consolidated four-fetch block in the 0121 entry
main action:         none — no merge or push is authorized
record landing:      this notes delta is committed on the record branch by this close turn
```

Resume state: unchanged from the 0121 entry — candidate frozen at
`a0a03a9`, audit pre-armed, transport user-owed via the four-fetch block;
`0126`/`0127` remain queued and pre-ruled (settled receipts; they close
with pointers on arrival, no reply owed); the next substantive Architect
work in this lane is the landings-2+3 audit (`0132-to-fable`).

## TEX-PROSE-04+05+06 closure receipt (0126) closed as PRE-RULED: the HOLD re-verified unchanged; no echo into the terminated loop (2026-07-14, Fable/Architect)

Inbound: mailbox `0126-to-fable`, the Red Team's routing copy of its own
RULING-A closure acknowledgment (register section "TEX-PROSE-04+05+06
RULING-A closure acknowledgment: exact tip remains frozen; no further Red
Team action (2026-07-14)", above).  This is exactly the arrival the 0121
adjudication pre-ruled: "`0126-to-fable` and `0127-to-fable` are settled
receipts of my own TEX-PROSE and tools-review/07+08 rulings whose register
sections already ride committed history — on arrival they close with
pointers, no reply owed."  That premise was verified rather than assumed:
`git show HEAD:notes/red-team-audit-and-didactics-2026-07-13.md` carries
the acknowledgment section (two heading/cross-reference hits), so the
substance is already committed history.

Field-by-field comparison of the routing copy against its register
section: same verdict receipt (ACCEPTED/no-strike on the 0112 publication
delta; the RULING-A `888272b7...` extraction debt closed by the
Architect's independent base reimplementation and mutation probe, the
earlier Red Team evidence remaining input rather than self-adjudication),
same no-action posture (no clone command, no gate replay, no
edit/amend/rebase/merge/publish/push of any TeX branch state), same
frozen tip `5546a0fd74d9536fdab42bfc8352411fb144752d`, same user-owed
transport with the post-fetch `rev-parse` as the binding tamper screen,
same RULING-B sequencing (04+05+06 lands first; no 07+08 rebase until
both TeX substance audits return and a fresh handoff authorizes it), same
no-merge/no-push boundary.  Nothing new; no strike, no debt.

Re-verified this turn — the one question a quick close still owes is
whether the user fetch landed between the acknowledgment and this
dispatch, which would flip this close into the pre-armed at-tip substance
audit:

1. `git show-ref --verify refs/heads/codex/tex-prose-04-06` — rc 128,
   ref absent.
2. `git cat-file -t 5546a0fd74d9536fdab42bfc8352411fb144752d` — rc 128,
   tip object absent from the shared store.
3. `git worktree list` — no `codex-tex-prose-04-06` path; the candidate
   still sits in the UNLINKED clone (the bilaterally confirmed
   no-agent-side-transport class).

The HOLD therefore stands exactly as ruled; the 04+05+06 substance audit
stays PRE-ARMED on the one-line trigger `codex/tex-prose-04-06 reachable
at 5546a0f`; RULING B is unchanged; the consolidated four-fetch user
block in the 0121 entry above is unchanged and remains the only open
action on this thread.

Loop discipline: the TEX-PROSE-04+05+06 mailbox loop TERMINATED at
`0128-to-sol`, a terminal receipt that "requests no reply and authorizes
no turn-consuming follow-up."  A to-sol echo of this close would be that
follow-up, so none is sent; the outbound routing summary goes to the USER
(`0137-to-user`) — the swept-receipts precedent as applied at the 0125
close (`0135-to-user`), and the Architect-side instance of the
terminal-vs-unconditional-preamble conflict Sol's `0133-to-fable` /
`0134-to-fable` route into the tools-review repair unit.  No `MEMORY.md`
edit: line 112 already records the thread termination and the
user-fetch-only open state, and stays accurate as written — which also
keeps this commit clear of the Implementer's uncommitted landings-2+3
files awaiting their own audit turn (`0132-to-fable`, the next
substantive work in this lane).

Sequence-number hygiene: the mailbox was re-listed immediately before
writing (the ruled mitigation for the live `next_seq()` collision
defect); the highest existing sequence at that instant was `0136`, so the
outbound was written as `0137-to-user`.

Landing block, printed only:

```text
record branch:       claude/amazing-keller-e798b6
record file:         notes/red-team-audit-and-didactics-2026-07-13.md
outbound transport:  notes/mailbox/0137-to-user.md
protected TeX tip:   codex/tex-prose-04-06 at 5546a0fd74d9536fdab42bfc8352411fb144752d
clone action:        none — the exact tip stays frozen
branch publication:  user runs the consolidated four-fetch block in the 0121 entry
main action:         none — no merge or push is authorized
record landing:      this notes delta is committed on the record branch by this close turn
```

Resume state: unchanged for the TeX thread — 04+05+06 tip frozen at
`5546a0f`, both TeX substance audits pre-armed, transport user-owed via
the four-fetch block; `0127` remains queued and pre-ruled (it closes with
a pointer to its bilateral-HOLD entry on arrival); the next substantive
Architect work in this lane is the landings-2+3 audit (`0132-to-fable`).

## Tools-review + TEX-PROSE-07+08 bilateral-HOLD receipt (0127) closed as PRE-RULED: both tips re-verified absent; no echo into the settled loop (2026-07-14, Fable/Architect)

Inbound: mailbox `0127-to-fable`, the Red Team's routing copy of its own
bilateral-HOLD acknowledgment (register section "Tools-review +
TEX-PROSE-07+08 bilateral-HOLD acknowledgment: exact tips remain frozen;
no further Red Team action (2026-07-14)", above).  This is the last of the
three arrivals the 0121 adjudication pre-ruled: "`0126-to-fable` and
`0127-to-fable` are settled receipts of my own TEX-PROSE and
tools-review/07+08 rulings whose register sections already ride committed
history — on arrival they close with pointers, no reply owed."  That
premise was verified rather than assumed:
`git show HEAD:notes/red-team-audit-and-didactics-2026-07-13.md` carries
the acknowledgment section (two heading/cross-reference hits), so the
substance is already committed history at `cb8345f`.

Probe-tooling hazard found while verifying (one line, so no future turn
repeats it): in this headless sandbox `git grep <pattern> <tree-ish> --
<path>` returns NO output even for patterns provably present in the
committed file — a silent false negative that would read as "section not
committed."  The `git show <tree-ish>:<path> | grep` form is the one that
answers truthfully; it is the form of record for committed-history checks.

Field-by-field comparison of the routing copy against its register
section: same verdict receipt (ACCEPTED, compliant phase-one handback, no
strike on publication delta 0113; the shared-repository absence checks and
the bilateral transport probes credited as the Architect's independent
evidence, the earlier Red Team transcripts remaining input rather than
self-adjudication), same no-action posture (neither blocked publication
re-attempted, neither isolated clone inspected or modified, no gate
replayed, no branch amended/rebased/merged/pushed), same frozen tips
byte-identical to the ruling (`codex/tools-review` at
`96e5f26a778f759b665292c1bb35c74ee17daf3c`; `codex/tex-prose-07-08` at
`f085260ee0df3097fa1438dbaff72251d0ef2205`), same user-owed transport from
both lanes with the post-fetch `rev-parse` as the binding tamper screen
and no further Red Team headless publication attempts for this class, same
audit routing (`codex/tools-review reachable at 96e5f26` re-opens the
tools audit; 07+08 pre-armed at the independently reproduced
`97e938bb...`), same RULING-B sequencing (04+05+06 lands first; no 07+08
rebase until both TeX substance audits return and a fresh Architect
handoff authorizes it), same reachability-authorizes-no-merge/no-push
boundary.  Nothing new; no strike, no debt.

Re-verified this turn — the one question a quick close still owes is
whether the user fetch landed between the acknowledgment and this
dispatch, which would flip this close into the pre-armed at-tip audits;
this thread carries TWO tips, so both were re-checked:

1. `git show-ref --verify refs/heads/codex/tools-review` — rc 128, ref
   absent.
2. `git show-ref --verify refs/heads/codex/tex-prose-07-08` — rc 128,
   ref absent.
3. `git cat-file -t 96e5f26a778f759b665292c1bb35c74ee17daf3c` — rc 128,
   tools-review tip object absent from the shared store.
4. `git cat-file -t f085260ee0df3097fa1438dbaff72251d0ef2205` — rc 128,
   07+08 tip object absent from the shared store.
5. `git worktree list` — neither a `codex-tools-review` nor a
   `codex-tex-prose-07-08` path appears; both sources remain UNLINKED
   clones (the bilaterally confirmed no-agent-side-transport class).

The HOLD therefore stands exactly as ruled; the tools audit stays
PRE-ARMED on the one-line trigger `codex/tools-review reachable at
96e5f26`; the 07+08 substance audit stays PRE-ARMED at the
Architect-reproduced `97e938bb...` preservation hash; RULING B is
unchanged; the consolidated four-fetch user block in the 0121 entry above
is unchanged and remains the only open action on this thread.

Loop discipline: this exchange closed bilaterally with Sol's own terminal
receipt — the 0127 acknowledgment "records receipt and preservation only"
and commits that "no further headless Red Team turn will be spent trying
to publish this class" — and the 0121 pre-ruling says receipts of this
class close "with pointers, no reply owed."  A to-sol echo would re-open a
settled exchange and hand Sol a turn with nothing in it, so none is sent;
the outbound routing summary goes to the USER (`0138-to-user`) — the
swept-receipts precedent as applied at the 0125 (`0135-to-user`) and 0126
(`0137-to-user`) closes, and a third Architect-side instance of the
terminal-vs-unconditional-preamble conflict Sol's
`0133-to-fable`/`0134-to-fable` route into the tools-review repair unit.
No `MEMORY.md` edit: line 113 already records the 0113 adjudication, the
bilateral HOLD, and the audit re-request trigger, and stays accurate as
written — which also keeps this commit clear of the Implementer's
uncommitted landings-2+3 files awaiting their own audit turn
(`0132-to-fable`, the next substantive work in this lane).

Sequence-number hygiene: the mailbox was re-listed immediately before
writing (the ruled mitigation for the live `next_seq()` collision
defect); the highest existing sequence at that instant was `0137`, so the
outbound was written as `0138-to-user`.

Landing block, printed only:

```text
record branch:       claude/amazing-keller-e798b6
record file:         notes/red-team-audit-and-didactics-2026-07-13.md
outbound transport:  notes/mailbox/0138-to-user.md
protected tips:      codex/tools-review at 96e5f26a778f759b665292c1bb35c74ee17daf3c
                     codex/tex-prose-07-08 at f085260ee0df3097fa1438dbaff72251d0ef2205
clone action:        none — both exact tips stay frozen
branch publication:  user runs the consolidated four-fetch block in the 0121 entry
main action:         none — no merge or push is authorized
record landing:      this notes delta is committed on the record branch by this close turn
```

Resume state: unchanged for the tools-review/07+08 thread — both tips
frozen, both at-tip audits pre-armed, transport user-owed via the
four-fetch block; all three pre-ruled arrivals (0125/0126/0127) are now
CLOSED, so the pre-ruling ledger for the terminated loops is empty; the
next substantive Architect work in this lane is the landings-2+3 audit
(`0132-to-fable`).

### Unit-13 covariance package — Red Team implementation return (2026-07-14)

The never-dispatched reassignment is now implemented at
`2fd8a9dcd816c2681b708406b40d7bd81b7270d3` on
`codex/unit-13-covariance`, based on main `f347b8f`. Register items 25M-08,
25M-11, 25M-12, and 45M-01 each have a covariance-owned pure witness with a
mutation or legacy false-green arm. The encompassing unit-13 schema and
lensing-range contract has its own fifth witness. All five pass; the board
selftest is ALL PASS; every touched Python file compiles; diff-check and the
no-comprehension/lambda/missing-docstring/over-90 scans are clean.

The complete files, exact commands, discriminating numerical outputs,
explicit YAML block, workstation limitation, no-self-certification line,
and printed fetch/landing block are in `families-scalar-cmb.md`, "Unit-13
covariance package — Red Team implementation readback". Architect audit is
required before any user merge. The branch tip is in an isolated clone
because the sandbox rejected the linked-worktree ref lock; the readback's
fetch block makes it reachable without rewriting the tip.

ARCHITECT ADJUDICATION (2026-07-14): substance GO. The tip was reachable
(the user's fetch had already published the ref), so the audit ran against
the real object: byte-faithful scratch extraction (206/206 blob ids
verified), all five witnesses re-run and reproduced with every
discriminating number byte-matched, board-selftest ALL PASS, independent
AST style scan clean, all four contracts (base unit-13 + 25M-08/11/12 +
45M-01) walked clause-by-clause against the diff and satisfied, gate
surface untouched, seam disjoint from the in-flight adapter half. ONE
REQUIRED DELTA from the Architect's own unscripted probes: removing any of
the four validator calls from main() leaves all five witnesses green — the
wiring is unproven (the unit-41 production-coupling class); main-driving
refusal legs are dispatched back to the red team as a rider on the same
ledger line. Merge HELD until the delta lands so main receives one squash
per audited unit. Torch cmb-identity and the real-CAMB Planck-LCDM
byte-identity control stay workstation-owed. Full record:
`families-scalar-cmb.md`, "Unit-13 covariance audit (Architect,
2026-07-14)".
Caption deletion (Second Implementer/Sol, 2026-07-14): in `texnotes/emulator_code_guide.tex`, deleted “A Doctor-inspired science traveler uses a sonic tool to assemble a cosmological surrogate from measured coordinates, transformations and checks.”; the frontispiece and technical caption remainder are unchanged.

Landing block (user only, after Architect audit):

```bash
git fetch /Users/vivianmiranda/data/COCOA/june2026/emulators_code_v2/.claude/worktrees/amazing-keller-e798b6/.claude/worktrees/codex-tex-caption-doctor-sol codex/tex-caption-doctor:codex/tex-caption-doctor
git switch main
git merge --ff-only codex/tex-caption-doctor
git push origin main
```

### Unit-13 covariance wiring delta — Red Team return (2026-07-14)

The one audit-held production-coupling delta is implemented at
`7583019f6408363fa28f46e6e8b4aaacf7075137`, directly above audited unit tip
`2fd8a9d`. Only `redteam_covariance_params_witness.py` changed. Its real-main
harness now proves the `cov_args`, params, post-signal PSD, and final output
finiteness calls load-bearing. All four entry-point refusal legs pass on the
unmutated producer; bypassing each corresponding call makes its owned leg
red with rc 1; the producer was restored to blob `f0819ae1...` afterward.
All five covariance witnesses return rc 0, board-selftest reports ALL PASS,
the touched file compiles, diff-check is clean, and the six-file AST/style
scan is empty. Full raw evidence and the two-phase fetch block are in
`families-scalar-cmb.md`, "Unit-13 covariance wiring delta — Red Team
implementation readback". The Torch and real-CAMB legs remain
workstation-owed. Architect audit is requested before merge.

This is a Red Team delta implementation return, not certification. Awaiting
Architect audit; the Red Team does not self-certify.

## Unit-94 substance audit (Architect, 2026-07-14): GO — every gate re-run at the published tip, all pre-armed numerics byte-matched, one unscripted production-coupling probe fired, seam clean against record HEAD

Trigger: the user fetch landed (`0155-to-fable`); this turn re-verified
`refs/heads/codex/unit94-boundary-interior` = `a0a03a9f06541eaa8dfbbb4968f53dacfe9d4849`
with parent exactly `204748e2389a079cbc0c70446a306a6daf9771a6` — the frozen
tip and base the 0121 adjudication pinned. One commit, four files:
`compute_data_vectors/generator_core.py` (+109/-6), NEW
`gates/checks/redteam_unit94_boundary_witness.py` (+1010), and the two
readbacks (register +124, `data-generation-and-cuts.md` +85). No existing
gate, threshold, fixture, or golden base is touched — the gate-integrity
screen is clean by construction.

### Every gate re-run by this turn (full tip tree extracted to scratch)

1. **Ordinary witness**: rc 0, **12 [PASS]**, terminal
   `uniform-boundary-witness: ALL PASS`. The H0 retention
   `0.9992369413375854`, the offset resolved support
   `[1000.0000610351562, 1000.0099487304688]`, and the refusal messages all
   BYTE-MATCH the 0121 pre-arm — including the f32-width-ratio pin (my
   independent f32 implementation produced these exact digits; the f64
   variant differs in the 8th digit, so the witness computes honestly).
2. **All four mutation arms red at the claimed counts**:
   endpoint-times-constant rc 1 / 4 failed arms (H0 retention
   `0.29950401186943054` byte-exact = the minting defect restored);
   request-validation-bypass rc 1 / 5; resolved-validation-bypass rc 1 / 1;
   sampling-before-resolution rc 1 / 7. Exactly the 4/5/1/7 of the return.
3. **py_compile**: both touched files compile clean.
4. **Board self-test at tip**: rc 0, **176 [PASS] / 0 [FAIL]** — reconciled:
   176 equals the 204748e base census (the 182 baseline belonged to the
   record branch's fixed-facts additions), and no `redteam_*` witness has
   ever been board-registered, so no wiring was owed. The witness is a
   standalone gate script per the unit-41/53 pattern.
5. **Unscripted production-coupling probe (mine, not scripted by Sol)**:
   mutated the PRODUCTION helper's lower resolution outward
   (`np.nextafter(low, low - 1.0)`) — the witness reds 5 arms (policy
   surface, H0 interior, offset support, negative mirror, f32-adjacent
   refusal) and restores green when reverted. The witness exercises the
   real production code, not a private copy.

### Contract clause-by-clause (the 25M-01 minting contract)

ONE named helper `resolve_uniform_sampling_support(names, bounds)` at
module level, policy `nextafter-toward-interval-interior-v1`, working in
interval coordinates via `np.nextafter(low, high)` / `np.nextafter(high,
low)` — the dispatch's preferred form verbatim. Finite, ordered,
representably-nonempty interior validated BEFORE sampling: every refusal
witness records **zero sampler calls**. The helper returns requested AND
resolved per-name support plus the policy name, exposed at the named
surface `self.uniform_sampling_support` before `rng.uniform` runs; the
production branch samples from the resolved bounds. The fourth `bounds`
key, flagged at 0121, is RULED an alias: it is the resolved array assigned
from the same loop variables as `resolved` — one fact, one owner; its
semantics are "the array form of `resolved`" and it must never be
authored independently. Seam respected: no dataset identity/manifest code
(unit 8's half), facts sidecar untouched (landing 1's surface),
`confidence=0.9999994` untouched. House style holds (formal Arguments
block, named parameters, explicit loop, no comprehensions).

Call-site check beyond the pre-arm: `__run_mcmc` is invoked exactly once
per GeneratorCore (from `__init__`, rank 0), so the
`self.bounds = support["bounds"]` overwrite cannot compound across calls;
a fresh process rebuilds `self.bounds` from config. The overwrite also
implements persist-resolved (never-trust-defaults): downstream consumers
of `self.bounds` see the support actually sampled.

### Seam and merge-cleanliness at record HEAD

At current record HEAD the endpoint-times-constant pre-image sits
byte-intact at `generator_core.py:1048-1049` (displaced +302 by the
fixed-facts landings), both insertion contexts (`# Free Functions` /
`capture_native_output`) are intact, and
`gates/checks/redteam_unit94_boundary_witness.py` does not exist — the
unit-94 hunks apply with ZERO textual conflicts, satisfying sequencing
clause 2 of the ruling-A entry.

### VERDICT: GO

Unit 94 is audited GO at `a0a03a9`. Consequences, executed this turn:

- backlog line unit 94 RETIRED (GO recorded in this commit, per the
  ledger-hygiene rule);
- unit 8 UNBLOCKS: its line updated and the re-dispatch sent to the
  Implementer lane (`0163-to-opus`), citing this entry as the landing
  record 0103's premise required. The dispatch orders the Implementer to
  first merge `a0a03a9` into the record branch (a branch merge, not main —
  within agent authority) expecting ZERO generator_core.py conflicts, stop
  and hand back on any conflict;
- the codex branch stays frozen at `a0a03a9` as the audited source of
  record.

Landing block (printed only — merges and pushes to main stay the user's):

```text
record branch:       claude/amazing-keller-e798b6
record file:         notes/red-team-audit-and-didactics-2026-07-13.md (this entry)
audited tip:         codex/unit94-boundary-interior at a0a03a9f06541eaa8dfbbb4968f53dacfe9d4849 (frozen)
branch action:       Implementer merges a0a03a9 into claude/amazing-keller-e798b6 (zero-conflict expectation) as step 1 of the unit-8 dispatch
main action:         USER ONLY, at the next landing window — one squash commit for unit 94:
                       git merge --squash a0a03a9   (from the main checkout)
                       commit message: plain-words description of the uniform-bounds fix
                       then merge main back into claude/amazing-keller-e798b6
```

Commit hygiene note: this notes-only commit necessarily carries two of
Sol's uncommitted notes-first return records appended earlier today to
shared note files (the unit-13 wiring-delta readback at `7583019` and the
tex-caption deletion record) — records awaiting their own adjudication,
named here per the pre-squash foreign-commit discipline; no code files are
swept.

No self-certification override: this GO is the Architect's independent
audit — every gate re-run by this turn's own hands, all pre-armed numerics
byte-matched, one unscripted probe beyond Sol's scripted arms.

### Unit 94 current-main landing verification (Codex recovery, 2026-07-14)

The user's temporary recovery authority was exercised after the frozen-source
walk showed exactly one branch-only commit, `a0a03a9`, and no foreign commit.
The production hunk, standalone witness, and home-note readback were ported to
current `main`; the branch's stale register append was discarded because this
newer independent GO entry already supersedes it.

Codex re-ran the complete CPU acceptance surface on the integrated tree:

```text
ordinary witness                         rc 0  12 PASS; ALL PASS
mutation endpoint-times-constant         rc 1  4 failed arms
mutation request-validation-bypass       rc 1  5 failed arms
mutation resolved-validation-bypass      rc 1  1 failed arm
mutation sampling-before-resolution      rc 1  7 failed arms
py_compile (production + witness)         rc 0
gates/board.py --self-test                rc 0
git diff --check                          rc 0
```

The positive numerics reproduce the audit pins exactly: retained H0 fraction
`0.9992369413375854` and offset support
`[1000.0000610351562, 1000.0099487304688]`. Every mutation reds only through
the expected acceptance surface. Unit 94 is therefore safe to finalize as one
main commit; unit 8 remains a separate open dependency consumer.
