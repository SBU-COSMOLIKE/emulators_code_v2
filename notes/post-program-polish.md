# Post-program polish: READMEs, doc deep pass, Alien-Python sweep (spec)

**Date:** 2026-07-10. **Status:** SPEC (Architect, Fable) — QUEUED as
unit 5, after GEO closes (the sweeps must run over the FINAL layout,
not files about to move). **Spec code:** POL. Three phases, executed
in order, each its own commit(s) and acceptance.

## The request (user directives, 2026-07-10)

1. Update the main README and `emulator/README.md` — the code map "seems
   outdated" (confirmed: last touched at GRF fb71a25; it predates FTW,
   TPE, SPE, and everything after — no warmstart.py, no inference.py,
   no scalar family, no cobaya_theory pointer, stale driver list).
2. A deep pass over ALL files' documentation: is it updated (accurate
   post-units), and is it AS DIDACTIC as the current README — the full
   rule set from the README campaign applied to docstrings.
3. An "Alien Python" sweep: simplify the Python wherever performance is
   not degraded. Explicit loops are fine when they cost nothing — the
   user is mainly a C coder.

## POL-1 — the READMEs

- Main README: a consolidation pass after the units — the per-unit
  sections land IN their units (SPE's section 14 shipped; CME/BSN carry
  README drafts per their specs); POL-1 checks coherence across them
  (cross-references, the Contents, the glossary, appendix 20's
  two-door examples growing CMB/BSN entries) and finishes the
  still-commissioned global didactic pass over any section not yet
  touched by the campaign rules.
- `emulator/README.md` (the code map): full refresh against the final
  tree — layout, what-each-file-does, the change-X-edit-Y table, and
  the per-file function appendices gain every post-GRF member
  (warmstart, inference, geometries/ after GEO, losses/scalar +
  transfer, designs additions, the new drivers, cobaya_theory/, the
  gates map); dead entries die. The code map follows the same didactic
  rules as the main README (it is a README).
- Acceptance: the anchor-verification script green on both files (the
  56-then-101-anchor precedent); every named file exists (a
  path-existence census script — no stale pointers).

## POL-2 — the documentation deep pass (all files)

File-by-file over emulator/, the drivers, cobaya_theory/,
compute_data_vectors/, gates/: every module header and public
docstring checked on TWO axes —
- **Accurate:** does it describe the file as it IS after the program
  (units, folders, new members, changed flows)? Stale claims die.
- **Didactic to the README's standard:** the campaign rule set applied
  to docstrings — define-or-drop jargon at the use site (audience:
  cosmologists, not AI experts); shape-flow diagrams with every symbol
  in a legend where data flows; "whiten" never bare; no clause-bearing
  parentheses; tables/lists where enumerations hide in prose. The
  gates/checks sweep (GBC Part 3) is the precedent and stays the
  floor — this pass extends the standard to the whole tree.
- ZERO logic change (doc-only diffs; py_compile + an AST body-hash
  census proving code untouched is the acceptance — the doc-only
  claim is verified, not asserted).

## POL-3 — the Alien-Python sweep

**The calibration (user):** mainly a C coder; simplify wherever
performance is NOT degraded; explicit loops are fine when they cost
nothing. [[py-module-style-conventions]] is the base (it already bans
comprehensions outside hot loops); this sweep hunts what remains:

| alien construct | the C-readable form |
|---|---|
| walrus (`:=`) | assign, then test |
| nested / conditional comprehensions | explicit loops with named intermediates |
| generator-expression chains | a loop building a named list |
| clever unpacking / starred gymnastics | indexed or named access |
| ternary pileups | if/else blocks |
| lambda-where-a-def-reads-better | a small named function |
| functools / itertools cleverness | the loop it abbreviates |
| `**kwargs` indirection where the call site can be explicit | named parameters (house style already) |

**The two hard rules:**
1. **Performance is never degraded.** Hot-path constructs (the
   training loop, einsum contractions, vectorized numpy/torch, the
   conv/attention math) KEEP their optimized forms — the fix there is
   a better comment, never a slowdown. Cold paths (config validation,
   staging glue, drivers, gates) prefer the C-readable form.
2. **Behavior-preserving, proven.** Every edit rides the board: full
   green + the golden byte-identity leg where configured; the
   Mac-side proof per file is an output-equivalence probe on the
   edited spans (exec old vs new on the same inputs) where feasible.
   Own commits, separate from POL-2's doc-only diffs (different risk
   classes, different review).

## Sequencing

Unit 5: after GEO. Internal order POL-1 -> POL-2 -> POL-3 (docs
stabilize before code edits re-touch files); POL-2 and POL-3 may share
one file-by-file walk but ship as separate commits per the risk split.

## Links

[[docs-plain-language-define-or-drop]], [[readme-yaml-needs-snippet]],
[[py-module-style-conventions]], [[gates-checks-docs-plain-language]],
[[designs-losses-family-folders]] (the GRF precedent for
byte-discipline), [[geometry-family-folder]].

## Resume state (Architect, 2026-07-11, overnight-mode continuation)

**POL-1 — partially executed; the remainder has hard sequencing gates:**

- DONE: `emulator/README.md` (the code map) fully rewritten against the
  final tree — the five-family table, the geometry folder + shims, all
  post-GRF members (background/syren_base/warmstart/inference/
  family_drivers, losses scalar/cmb/transfer, the 14-driver table, the
  cobaya_theory and compute_data_vectors maps, the gates section), the
  refreshed change-X table (incl. the "a NEW output family" recipe row),
  the Variants table grown by Transfer + Fine-tuning rows. Acceptance
  ran: the anchor census (6/6 links) and the path-existence census
  (79/79 backticked paths resolve) both green.
- BLOCKED-ON-MERGE (recorded, deliberate): the MAIN README
  consolidation. The main checkout's README diverges from this branch
  by design (the board-green-queue memory), and THREE section drafts
  wait in notes/ (readme-cmb-section-draft, readme-baosn-section-draft,
  readme-mps-section-draft + its Drivers table). Order of operations:
  the user merges origin/main, the drafts land in the MAIN checkout,
  THEN the POL-1 coherence pass (cross-references, Contents, glossary,
  appendix 20's new family examples) runs over the merged document.
- DEFERRED-UNTIL-BOARD-GREEN (recorded ruling): the cosmic-shear driver
  renames into the `<verb>_<family>_emulator.py` namespace. Renames
  touch the board configs; landing them under a never-run 32-gate board
  would make a red board undebuggable. They land AFTER the first full
  32-gate green, as their own commit.

**POL-2 / POL-3 — queued (the next session's work), inventory:**
the tree at execution = emulator/ (20 modules + 3 family folders),
5 cobaya_theory adapters, 6 compute_data_vectors files, 14 drivers,
gates/ (board + 18 checks). POL-2 walks it file-by-file (accuracy +
the README didactic standard; doc-only, proven by the AST body-hash
census). POL-3 hunts the alien-construct table (hot paths never
slowed; every edit an output-equivalence probe + the board). The two
passes may share one walk but ship as separate commits (the risk
split). NOTE: most files touched by the five 2026-07 units were
WRITTEN under the standard already; the deep pass's real targets are
the pre-SPE tree (training.py, batching.py, scheduling.py,
analytics.py, activations.py, designs/, losses/core+ia+pce,
plotting.py, the four cosmic-shear drivers, gates/board.py's older
gates).

## Resume state (Architect, 2026-07-11 second pass — POL executed)

The user directed the remaining queue to run without waiting for the
merge/board ("continue do the tasks"); everything below is uncommitted
in the amazing-keller worktree, landing blocks printed in the session
close.

**POL-1 main README — DONE (the wait-for-merge gate was replaced by a
proof).** The worktree README.md = main's README (git show main:) +
the three family sections (15 CMB / 16 BSN / 17 MPS from the notes/
drafts) + the Drivers subsection at the end of section 1
(anchor `drivers-table`) + the EMUL2 pointer paragraph in section 1 +
the transfer-scope paragraph in section 13 + appendix 23 grown
(five-adapter intro/diagram, three predict-returns rows, the
background scripting pattern) + appendices renumbered 15..21 ->
18..24 (descending replace) + two coherence catches fixed (the
glossary sentence whose link text still said 14/15, and six
`geometries_*.py` references updated to the post-GEO
`geometries/*.py` paths, length-neutral in the ASCII diagrams).
Acceptance: anchor census 113/113 links + path census green on both
READMEs (the probe needed contextual-root + fenced-block fixes, both
probe bugs). MERGE FACT, proven with git merge-file: README.md
CONFLICTS on merge (main's appendix-20 commit touched the same lines
the renumber moved) and the resolution is always the branch side —
`git merge-file --ours` output is byte-identical to the consolidated
file, so nothing from main is lost. Resolution command for the user,
merging the branch into main:
`git checkout --theirs README.md && git add README.md`.

**EMUL2 acceptance YAML — ships as
`cobaya_theory/EXAMPLE_EMUL2_EVALUATE.yaml`.** Mirrors
projects/lsst_y1/EXAMPLE_EMUL2_EVALUATE1.yaml (read from the june2026
Cocoa checkout) with the three legacy theory blocks (emulrdrag GP /
emulbaosn pt+extrapar / emulmps keras) replaced by emul_scalars +
emul_baosn + emul_mps; likelihood, params, and the evaluate override
kept so the two runs evaluate the same point. The emulators lists
carry placeholder roots (projects/lsst_y1/emulators/{rdrag,baosn,mps}/
...) for the user to point at trained artifacts.

**POL-2 — DONE (doc-only, proven).** Stale-claim fixes:
plotting.py header (family pages added), losses/core.py (subclass
list + scalar.py/cmb.py), designs/plain.py (family-blind trunk
sentence + the geometry paths in the diagram to slash form),
training.py (loss block is {mode, berhu, roughness}),
train_single_emulator_cosmic_shear.py (the driver trains cmb/grid/
grid2d too — matches the new Drivers table). The rest of the
inventory (batching, scheduling, analytics, activations, blocks,
designs/pce, losses/ia+pce, the other three cosmic-shear drivers)
audited ACCURATE as-is — the GBC-era campaign already ran
define-or-drop over them. PROOF: the AST body-hash census
(scratchpad ast_body_hash.py) over all 88 repo .py files is
byte-identical before/after the doc pass.

**POL-3 — DONE (8 files, probes 7/7).** No walrus anywhere; single
ternaries KEPT (C has ?:, the table bans pileups; none nested);
`nn.Sequential(*layers)`-style stars KEPT (idiomatic, not
gymnastics); plotting's itertools.cycle and pce's
itertools.combinations KEPT (conventional/mathematical). Converted,
all cold paths: activations.py 5 factory lambdas -> named factories;
designs/blocks.py 2 factory lambdas -> module-level affine_norm /
identity_norm + the ResBlock default-arg lambda -> affine_norm;
analytics.py 2 assigned coerce lambdas -> defs; training.py 3
conditional dict comprehensions -> explicit loops (the roughness
strip; opt_extras/sched_kwargs hoisted above resolved_train);
the 4 min(key=lambda) epoch picks -> named epoch_rank defs (train
driver, tune driver x2, family_drivers); plotting.py
`coefs, *_ = lstsq` -> `[0]` indexing. PROOF: probe_pol3.py 7/7
(old spans exec'd from git HEAD vs new, same inputs -> identical
outputs, including the ValueError texts and the ResBlock default);
the AST census confirms the code-changed set is EXACTLY these 8
files; full compileall green. Torch-side confirmation rides the
normal board (no gate config touched).

**Still open, unchanged rulings:** the driver renames stay
DEFERRED-UNTIL-BOARD-GREEN; MPS-DIAG (grid2d diagnostics pages) and
the other recorded interims ride later deltas.

## The driver renames — EXECUTED (user order, 2026-07-11)

Two rulings changed by explicit user instruction, superseding the
records above:

1. **The namespace is FAMILY-FIRST** — `<family>_<verb>_emulator.py`,
   "what you are emulating comes first always" — superseding D-MP5's
   verb-first `<verb>_<family>_emulator.py`. (Bonus: `ls` now groups
   the drivers by family.)
2. **Executed BEFORE the first 32-gate board run** — the
   deferred-until-board-green ruling was the Architect's caution; the
   user ordered the rename now, after proving main in sync.

The map (17 drivers, `git mv` + a longest-first text sweep — needed
because sweep_ntrain_scalar_emulator CONTAINS train_scalar_emulator):

    train_single_emulator_cosmic_shear  -> cosmic_shear_train_emulator
    train_{scalar,cmb,baosn,mps}_emulator -> {f}_train_emulator
    sweep_ntrain_emulator_cosmic_shear  -> cosmic_shear_sweep_ntrain_emulator
    sweep_ntrain_{f}_emulator           -> {f}_sweep_ntrain_emulator
    tune_single_emulator_cosmic_shear   -> cosmic_shear_tune_emulator
    tune_{f}_emulator                   -> {f}_tune_emulator
    bakeoff_activation_emulator_cosmic_shear
                                -> cosmic_shear_bakeoff_activation_emulator
    sweep_hyperparam_emulator_cosmic_shear
                                -> cosmic_shear_sweep_hyperparam_emulator

"single" was DROPPED (the family names never carried it). Five
driver-named example YAMLs moved with their drivers
(cosmic_shear_{train,tune,sweep_hyperparam,finetune,transfer}_
emulator.yaml). Rode along: gates/run_board.py `_DRIVER`,
board_config.json's golden-config yaml name + key description, the
FAMILY_DRIVERS map + wrapper imports/progs, the Optuna
`STUDY_NAME` "tune_single" -> "cosmic_shear_tune" (only test studies
exist, per the user), a ge_c_eval_bs print label, README shorthand
labels (`tune_single` -> `tune`; the appendix driver diagram relabeled
with column alignment kept), and both READMEs' tables/globs/examples.
notes/ history untouched by design — this section is the map.

Evidence: 20 `git mv` + 32 files text-swept (163 hits) + residuals;
leftover-name census clean outside notes/; full compileall; both
README censuses green (the path census re-proves every renamed
backticked path resolves); board registry 32; wrappers re-probed
(new prog/family pairs + the renamed engine import).

WORKSTATION NOTE: if a local board_config.json override exists there,
it must be dropped (`git checkout -- gates/board_config.json`) or its
golden-config yaml name updated — the repo copy now says
cosmic_shear_train_emulator.yaml.

## Generating-the-training-set promoted to a section (user, 2026-07-11)

"Appendix: Generating the training set" is generation — a pipeline
STAGE, not reference material — so it moved out of the appendix run:
now **section 18**, physically between the family sections and the
appendices (which shifted 19..22; scripting 23 and AI-Usage 24 kept
their numbers). While promoting, the pre-CME staleness died: the
section opened as if dataset_generator_lensing.py were the only
generator; it now opens with the four-generator table (lensing / cmb /
background / mps: family, truth code, file store), names
compute_cmb_covariance.py as the fifth generation-side tool (the
user's discoverability complaint — the script lives in
compute_data_vectors/, section 15 has its physics), frames the shared
core (generator_core.py) as what the rest of the section describes,
and generalizes the output-contract table's `_cs_` tags to
`_<probe>_`. Anchor census 113/113 + path census 39 paths green after
the move.
