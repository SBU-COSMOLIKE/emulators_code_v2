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
