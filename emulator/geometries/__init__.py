"""The geometry family folder (GEO unit).

One module per geometry kind: parameter (input whitening), output (the
cosmolike data-vector geometry), scalar (derived parameters), cmb (the
per-multipole diagonal), grid (background functions of z), grid2d (the
matter-power (z, k) surfaces). Import the submodule you need
(from emulator.geometries.scalar import ScalarGeometry); this __init__
deliberately re-exports nothing, so the import census stays exact.

The old flat modules (emulator/geometries_<name>.py) remain as shim
re-exports FOREVER: every schema-v2 artifact saved before this folder
existed persists its geometry classes under the old module paths, and
rebuild_emulator dispatches by importing exactly that stored string.
"""
