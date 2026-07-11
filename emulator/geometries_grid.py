"""Legacy shim: this module moved to emulator/geometries/grid.py
(the GEO unit). The shim stays because schema-v2 artifacts persist
geometry classes as FULL module paths and rebuild_emulator imports
exactly the stored string; new code imports the new home.
"""

from .geometries.grid import TARGET_LAWS, GridGeometry  # noqa: F401
