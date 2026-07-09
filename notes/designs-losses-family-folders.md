---
name: designs-losses-family-folders
description: "SPEC 2026-07-08 (Architect; user-proposed): gather every variant of the two family files into two child folders — emulator/designs/ {blocks, plain, ia, pce} and emulator/losses/ {core, ia, pce} — killing the three-files-same-name ambiguity and the IA/ + PCE/ one-purpose folders (plus the stale empty parallel/). Verbatim moves only, no style retrofit. Artifact-immune: .emul = state_dict tensors (no class pickles), h5 cls markers name only the flat geometry modules, PCE rebuild is a hardcoded import. activations.py stays flat (gsv drift-proof monkeypatches its path). Acceptance = fresh full board green; the ema-off-identity golden leg pins output byte-identity."
metadata:
  node_type: memory
  type: project
---

# designs/ + losses/: the family folders (spec, 2026-07-08)

User-proposed reorganization (superseding the same-day flatten-PCE
question): instead of merging the variant files into the flat twins,
gather every version of the two family files under a child folder of
`emulator/`, named by the family. The user asked for the designs
family explicitly; the twin losses folder is the Architect's
recommended companion (accepted with "make the plan") because after
the designs move, `IA/` and `PCE/` would each hold a single
`loss_functions.py` — one-file folders, worse than today.

## Why "designs"

It is the repo's own word, not an import: the base class is
`DesignSpec` (emulator_designs.py:55), the registry dict is
`IA_DESIGNS`, docstrings say "the factored design". Rejected:
`models/` (collides with the YAML `model:` block, `nn.Module`, and
cobaya's `Model`), `architectures/` (what `model.name` selects — 
narrower than the folder, which also holds shared blocks),
`networks/` (vocabulary the repo never uses). File names inside reuse
the exact plain / ia / pce labels the parity and save gates print.

## Target layout

```
emulator/designs/
  __init__.py     one-paragraph family map (which variant lives where)
  blocks.py       was emulator_designs_building_blocks.py
                  (Affine, BinLinear, make_norm, the CNN/TRF blocks)
  plain.py        was emulator_designs.py
                  (DesignSpec, ResMLP, ResCNN, ResTRF)
  ia.py           was IA/emulator_designs.py
                  (TemplateMLP, TemplateResCNN, TemplateResTRF)
  pce.py          was PCE/emulator_designs.py (PCEEmulator)

emulator/losses/
  __init__.py     the same map for the loss variants
  core.py         was loss_functions.py
                  (make_chi2, CosmolikeChi2, berhu family, anneal_value)
  ia.py           was IA/loss_functions.py
                  (TemplateFactoredChi2, nla_coeffs, ...)
  pce.py          was PCE/loss_functions.py
                  (PCEResidualChi2, PCERatioChi2)
```

Deleted afterward: `emulator/IA/`, `emulator/PCE/` (their `__init__`
rationale docstrings survive — the factored-IA paragraph into
`designs/ia.py`'s header, the deprioritized-for-xi paragraph + the
npce-and-ia-template-factoring note pointer into `designs/pce.py`'s
header), and the stale `emulator/parallel/` (holds only `__pycache__`;
its modules were absorbed into the flat tree long ago).

**`activations.py` stays flat and untouched.** It is shared beyond the
designs family (results.py rebuild, two gate check scripts), and the
save-rebuild-drift drift proof monkeypatches
`emulator.activations.make_activation` by that path — moving it buys
nothing the user asked for and risks the drift-proof's target.

## Why the move is artifact-safe (verified, not assumed)

- `.emul` = `torch.save` of the state_dict alone, every tensor cpu
  (results.py:145-224) — a name -> tensor mapping, no pickled classes,
  so no stored module paths.
- The h5 `cls` markers (rebuild's class dispatch) name only the flat
  geometry modules (`geometries_parameter.py` / `geometries_output.py`),
  which do not move.
- The pce group rebuild is a hardcoded import (results.py:459), not a
  stored path.
Every existing saved emulator therefore loads unchanged; the code move
is purely source-level.

## Complete import inventory (every touch point, swept 2026-07-08)

Designs family:
- emulator/experiment.py:71 (plain: ResMLP, ResCNN, ResTRF), :72-73
  (IA templates), :77 (make_norm from blocks); the MODELS registry
  values are these imports, so no further registry change.
- emulator/training.py:47 (ResMLP), :48 (Affine, BinLinear from blocks)
- emulator/results.py:395 (make_norm from blocks), :459 (PCEEmulator,
  function-local)
- gates/checks/gwd_census.py:32 (blocks imports)
- Intra-family: plain.py:51 and ia.py:24 import the blocks module
  (become `from .blocks import ...`).

Losses family:
- emulator/experiment.py:70 (make_chi2), :74-75 (IA), :1511-1512
  (PCE pair, function-local; :1511 also imports PCEEmulator — designs)
- emulator/training.py:49 (anneal_value)
- emulator/inference.py:169 (TemplateFactoredChi2), :181 (PCE pair)
- gates/checks/gb_c_berhu_reduce.py:31 (CosmolikeChi2)

Harness + docs:
- gates/board.py:262 preflight import line ->
  `import emulator, emulator.designs, emulator.losses`
- README.md: the pipeline-diagram and prose mentions (lines ~900, 906,
  1009, 1016, 1048) + the per-file appendix sections for all six moved
  files; sweep every mention, do not spot-fix.
- The moved files may carry further relative imports between
  themselves (e.g. a loss variant importing its design twin): the rule
  is intra-folder `from .x import`, cross-folder
  `from ..designs.x import` / `from ..losses.x import`; the
  Implementer sweeps every `from .` / `from ..` line in the six files.

## Discipline

Verbatim moves ONLY: class/function bodies byte-identical to their
sources; the only permitted edits are the import lines, the module
docstring headers (family context + preserved rationale paragraphs),
and the two new `__init__.py` maps. No style retrofit, no renames of
classes or functions, no doc-compression — any of that is a separate
unit. The ema-off-identity golden byte-identity leg is the numerics
pin: a refactor that changes any epoch line has a bug.

## Gates

- GRF-A (Mac, static): py_compile + compileall over the whole tree;
  zero surviving references to the retired paths (grep
  `emulator_designs`, `loss_functions`, `emulator.IA`, `emulator.PCE`,
  `IA/`, `PCE/`, `parallel/` across *.py, README, gates/ — hits allowed
  only in notes/, which are history); AST verbatim-move check (each
  moved class/def's source segment byte-equal to its origin, imports
  and headers excepted).
- GRF-B (workstation, user-run): a FRESH full board — move
  board_status.json aside so resume does not skip (every gate imports
  the package, so every gate re-validates; total gate time is ~10 min
  by run-10 timestamps). Acceptance = 18/18 PASS again, with
  ema-off-identity's golden leg proving output byte-identity and
  save-rebuild-drift + cobaya-adapter proving the artifact path.

## Status

SPEC 2026-07-08, handoff issued the same day. (Sequenced after BOARD
GREEN run 10; independent of the gates/checks plain-language docs
sweep, which touches gates/checks docstrings only.)

### Implementer GRF execution (2026-07-08, Opus, base 4734139)

IMPLEMENTED, uncommitted, GRF-A green. ff-synced the worktree from
eaeb383 to origin/main 4734139 (the spec commit) before starting.

Footprint (git diff HEAD --stat = 20 files, +151/-123, plus 2
untracked family __init__): seven `git mv` renames (git recorded all as
renames, history preserved) — building_blocks -> designs/blocks.py,
emulator_designs.py -> designs/plain.py, IA/emulator_designs.py ->
designs/ia.py, PCE/emulator_designs.py -> designs/pce.py,
loss_functions.py -> losses/core.py, IA/loss_functions.py ->
losses/ia.py, PCE/loss_functions.py -> losses/pce.py; two new maps
(designs/__init__.py, losses/__init__.py, untracked -> git add);
deleted IA/__init__.py + PCE/__init__.py (their rationale folded into
designs/ia.py and designs/pce.py headers). parallel/ needed NO deletion
(already gone since 29b23dd; not tracked, not on disk).

Moved-file edits = headers + imports ONLY (bodies byte-identical, proven
below). Intra-folder imports rewritten `.blocks` / `.plain` / `.core`;
cross-folder `..activations` / `..analytics` / `..geometries_output`
(all one level up from the family folder). External sites swept:
experiment.py (5 top imports + 2 func-local + 4 comment path-refs),
training.py (3 imports + 1 docstring), results.py (2 func-local),
inference.py (2 func-local), data_staging.py + geometries_output.py (1
docstring path-ref each), board.py:262 preflight -> `import emulator,
emulator.designs, emulator.losses` (+ its stale parallel comment
rewritten), gwd_census.py + gb_c_berhu_reduce.py (1 import each). Both
READMEs restructured: root diagram + build-steps (5 sites); code map =
new designs/ + losses/ family entries in the layout, the What-each-file
and Change-X tables, §4 Variants reframed to design+loss pairs, and the
appendix grouped into `apx-designs` (blocks/plain/ia/pce) + `apx-losses`
(core/ia/pce), old `apx-pce`/`apx-ia` sections folded in and removed.
The four "former emulator IA/PCE subpackage" provenance lines drop the
trailing slash so the `IA/`/`PCE/` grep stays clean; the §4 `parallel/`
history mention reworded (the folder token retired everywhere but
notes/). activations.py, the geometry modules, the schema, and every
gate config were left untouched; no class/function renamed.

GRF-A (Mac) GREEN: (1) full-tree token grep across *.py + *.md + *.yaml
+ *.json = ZERO surviving `emulator_designs` / `loss_functions` /
`emulator.IA` / `emulator.PCE` / `IA/` / `PCE/` / `parallel/` /
`building_blocks` (notes/ excluded as history); (2) py_compile of all 39
tracked .py OK + compileall emulator/ + gates/ clean; (3) AST
verbatim-move check ALL 7 pairs code_identical=True AND
top_names_identical=True (docstring- and import-stripped ast.dump vs the
pristine `git show HEAD:<oldpath>`); (4) bonus static import resolver —
all 118 emulator-internal imported symbols resolve to a real top-level
definition (a stand-in for the real import, no torch on the Mac).

GRF-B is the WORKSTATION acceptance (user-run): a FRESH full board with
gates/logs/board_status.json moved aside so resume skips nothing;
18/18 PASS, with ema-off-identity's golden leg proving output
byte-identity and save-rebuild-drift + cobaya-adapter proving the
artifact path end to end.

### 2026-07-08 — Architect: GRF audit VERDICT (worktree amazing-keller, base 4734139)
VERIFIED, commit-ready. Audited against the raw diff with an
independent verbatim-move probe (every top-level class/def/constant of
the seven renamed files byte-compared against `git show HEAD:<old>`,
docstrings INCLUDED — stricter than the Implementer's docstring-
stripped check). 5/7 files byte-identical outright; the two deltas are
both inside the spec's permitted-edits list and were re-proven
harmless: TemplateMLP (designs/ia.py) = one docstring path reference
the token-grep gate required (executable AST with docstrings blanked:
identical), and make_chi2 (losses/core.py) = a FUNCTION-LOCAL relative
import gaining one dot (`from .geometries_output` ->
`from ..geometries_output`) because core.py sits a level deeper — the
exact `from .`/`from ..` sweep the handoff mandated; same symbol, same
module, behavior identical. Token grep (Architect's own, corrected
exclusions): zero retired-path references outside notes/. compileall
clean. Every non-moved diff line is an import path or a provenance
comment; paren alignment preserved on the re-wrapped imports. Family
__init__ maps read plainly and teach the import style; the PCE
deprioritized-for-xi verdict + note pointer and the factored-IA
rationale both survive as the new module headers. README anchors:
apx-pce/apx-ia gone, apx-designs/apx-losses wired in TOC and sections.
No caps regressions in code/README added lines. activations.py, the
geometry modules, the schema, and every gate config untouched, as
mandated. GRF-B (workstation, user-run) remains the acceptance: fresh
full board (board_status.json moved aside), 18/18, the golden leg
pinning output byte-identity. Sequencing (corrected after the user
challenged it): the GCT-C MCMC smoke may run before OR after this
merge — the pre-merge ordering was one-variable-at-a-time caution, but
the refactor is byte-proven, the fresh board re-validates the adapter
chain anyway, and an import break vs a sampler break are trivially
distinguishable from the traceback; running the smoke on the
post-merge tree is if anything more representative. The only hard
requirement is that the FRESH full board runs after the merge.
