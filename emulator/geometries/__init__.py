"""The geometry family folder.

One module per geometry kind: parameter (input whitening), output (the
cosmolike data-vector geometry), scalar (derived parameters), cmb (the
per-multipole diagonal), grid (background functions of z), grid2d (the
matter-power (z, k) surfaces). Import the submodule you need
(from emulator.geometries.scalar import ScalarGeometry); this __init__
deliberately re-exports nothing, so the import census stays exact.

Saved artifacts persist geometry classes as full module paths, and
rebuild_emulator imports exactly the stored string, so every class
lives at its one canonical path in this folder. An artifact naming a
path that does not exist fails its rebuild with ModuleNotFoundError
naming the path — the geo-paths gate pins that loud death.
"""
