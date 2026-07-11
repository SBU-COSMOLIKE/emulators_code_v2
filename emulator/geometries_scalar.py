"""Legacy shim: this module moved to emulator/geometries/scalar.py
(the GEO unit). The shim stays because schema-v2 artifacts persist
geometry classes as FULL module paths and rebuild_emulator imports
exactly the stored string; new code imports the new home.
"""

from .geometries.scalar import ScalarGeometry  # noqa: F401
