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

### Durable-record rule

Every future Red Team handoff is appended to an existing topic or audit note
before, or in the same turn as, its chat copy.  The final handoff cites that
file and commit.  Chat history is never the sole copy of an Implementer
contract.
