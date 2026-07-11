"""Legacy shim: this module moved to emulator/geometries/parameter.py
(the GEO unit). The shim stays because schema-v2 artifacts persist
geometry classes as FULL module paths and rebuild_emulator imports
exactly the stored string; new code imports the new home.
"""

from .geometries.parameter import ParamGeometry, LogParamGeometry, AmplitudeFactorGeometry  # noqa: F401
