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
