"""Legacy shim: this module moved to emulator/geometries/output.py
(the GEO unit). The shim stays because schema-v2 artifacts persist
geometry classes as FULL module paths and rebuild_emulator imports
exactly the stored string; new code imports the new home.
"""

from .geometries.output import DataVectorGeometry, DiagonalGeometry, BlockDiagonalGeometry, build_shear_angle_map  # noqa: F401
