# Red-team audit of the landed batch and didactic repair queue (2026-07-13)

This is the Red Team's independent review of the Implementer's handoff at
`bd60a9f`, rechecked from the current merged tree at `a93f417`.  It is a
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

An untruncated scan of `emulator/` and `gates/` found 48 lines in 16 Python
files containing internal unit numbers, dated rulings, or reviewer biography.
Examples include `unit 60`, `unit 14(f)`, “Architect confirms,” and dated
“symmetry ruling” comments.  The prior cleanup searched only for the literal
`45M` token and therefore did not satisfy the no-internal-bookkeeping rule.

Required cleanup:

- Replace every internal code, queue number, reviewer name, and adjudication
  anecdote in Python docstrings, comments, errors, report labels, and board
  descriptions with the current mechanism and reason.  Identifiers required
  by executable compatibility may remain as identifiers, but human prose
  around them must use plain language.
- Completion is an untruncated zero-hit scan over the full pattern family,
  not only `45M`.  Record the exact patterns and scope in the gate output.
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

## Ordered Implementer queue from this review

1. Resume trust: make the raw-log digest part of the skip verdict; unify and
   expose executable/input manifests; extend preflight watch coverage.
2. Evidence rollout: apply the rulings above, then reconcile declared and
   executed assertion ids across all gates and check scripts.
3. Staging truth: make the seeded training/minibatch order independent of the
   RAM branch, correct the banner, define equality policy, and prove the real
   downstream sequence in both storage regimes.
4. Public numeric schema: remove optimizer and CMB reference-value coercions;
   complete the post-step finite workstation gate.
5. Workstation evidence already owed: generator replay/worker invariance,
   live scalar fine-tune parity, artifact save/forge/rebuild, and the full
   40-gate board.
6. README current-state repair, followed by the in-file teaching campaign in
   the priority order above.  Documentation-only batches carry AST-identity,
   compile, and untruncated stale-pattern evidence.

Do not mark any item closed from a source-text census alone when the claimed
behavior has an executable public boundary.  The landing handoff must name
the real path run, the mutation that the gate catches, and any capability lane
that remains unavailable.
