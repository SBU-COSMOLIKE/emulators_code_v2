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
