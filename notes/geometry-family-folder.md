# Geometry family folder (spec)

**Date:** 2026-07-10. **Status:** SPEC (Architect, Fable) — QUEUED as
unit 4, after BSN closes ("once we finish all these emulators", the
user, 2026-07-10). **Spec code:** GEO. **Home note** for the gate legs
riding the full-board acceptance.

## The request (user design goal)

The geometry files crowd the emulator/ root and keep multiplying —
geometries_parameter.py, geometries_output.py, geometries_scalar.py
today, plus CME's from-fiducial constructor + amplitude-law registry
and BSN's GridGeometry + target-law registry by the time this unit
runs. Gather them into a `geometries/` child folder exactly like the
GRF precedent did for designs/ and losses/
([[designs-losses-family-folders]]).

## The one hard constraint (why this is a unit, not a chore)

Every schema-v2 artifact persists its geometry classes as FULL MODULE
PATHS — `cls = "emulator.geometries_scalar.ScalarGeometry"` etc. — and
rebuild_emulator dispatches by importing that exact path. GRF recorded
itself "artifact-immune" precisely BECAUSE it left these modules flat;
this unit moves exactly the modules every saved artifact names. A
naive move breaks rebuild for every artifact ever saved.

**Ruling (mechanism, refinable at execution): legacy shim modules.**
The old flat files become 3-line re-exports
(`from .geometries.scalar import ScalarGeometry` under the old
module name), so:
- every OLD artifact's stored cls path imports forever (the marker
  still dispatches exactly — through an import alias to the one class
  object; isinstance stays sound);
- every NEW save writes the NEW path automatically
  (type(geom).__module__ materializes the new home at write time —
  the resolved-values rule doing the work);
- any user script importing the old paths keeps running.
A rebuild-side translation map was considered and rejected: shims
serve external importers too and touch zero rebuild logic.

## Design rules

- **D-GEO1 — layout (PROPOSAL, finalized at execution per the
  propose-don't-guess rule):** `emulator/geometries/{__init__.py,
  parameter.py, output.py, scalar.py, grid.py}` + the law registries'
  home decided by where CME/BSN actually put them. Verbatim moves
  only — no style retrofit, no logic edits (the GRF discipline).
- **D-GEO2 — the shims:** one flat legacy module per moved file,
  re-export only, each carrying a one-line docstring naming the new
  home and the reason (persisted cls paths). New code never imports
  the shims (a tree-wide census asserts it).
- **D-GEO3 — the import rewrite:** all current importers (13 files at
  spec time: the drivers, gates/board.py, three gate checks,
  warmstart/results/experiment/data_staging, designs/blocks +
  designs/plain, losses/core + losses/scalar) move to the new paths;
  the census re-runs at execution since CME/BSN will have grown the
  list.
- **D-GEO4 — gates:** (a) an old-path fixture leg — an h5 whose
  dv_geometry/param_geometry cls attrs carry the OLD flat paths must
  rebuild and predict (the artifact-compat proof, riding
  save-rebuild-drift or its own check); (b) a new-save leg — a fresh
  save writes NEW-path cls markers; (c) the shim-import census (new
  code clean of shims); (d) acceptance = fresh FULL board green (every
  gate touches geometries), with the ema-off-identity golden leg
  pinning byte-identity exactly as GRF's acceptance did.

## Out of scope (recorded)

Renaming classes (paths change, names do not); merging or splitting
geometry classes; retiring the shims (a far-future major version's
decision, recorded here so it is deliberate).

## Sequencing

After BSN closes, before the science thread — the natural hygiene
window, with all five-plus geometry members finally in existence so
the folder is cut once.

## Links

[[designs-losses-family-folders]] (the precedent + its acceptance
pattern), [[scalar-parameter-emulators]], [[cmb-spectra-emulators]],
[[baosn-emulators]].

## Resume state (Implementer appends below)

(not started — queued behind CME and BSN)
