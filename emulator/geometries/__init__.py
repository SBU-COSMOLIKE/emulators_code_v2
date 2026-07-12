"""The geometry family folder (GEO unit).

One module per geometry kind: parameter (input whitening), output (the
cosmolike data-vector geometry), scalar (derived parameters), cmb (the
per-multipole diagonal), grid (background functions of z), grid2d (the
matter-power (z, k) surfaces). Import the submodule you need
(from emulator.geometries.scalar import ScalarGeometry); this __init__
deliberately re-exports nothing, so the import census stays exact.

The old flat modules (emulator/geometries_<name>.py) are GONE. They
briefly lived as shim re-exports because schema-v2 artifacts persist
geometry classes as full module paths and rebuild_emulator imports
exactly the stored string; the user ruled (2026-07-11) that no
science artifact predates the folder, so the shims were retired. An
artifact carrying an old flat path fails its rebuild with
ModuleNotFoundError naming the path — the geo-paths gate pins that
loud death.
"""
