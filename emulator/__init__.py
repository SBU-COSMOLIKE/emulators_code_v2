"""Training, persistence, and prediction for the CoCoA SONIC emulators.

The package supports five observable families. Cosmic-shear data vectors use
the masked covariance geometry in ``geometries/output.py`` and the losses in
``losses/core.py``. Scalar outputs use ``geometries/scalar.py``. CMB spectra
use ``geometries/cmb.py`` and ``losses/cmb.py``. Background functions use
``geometries/grid.py``. Matter-power surfaces use ``geometries/grid2d.py``.
``experiment.py`` selects one of these owners from the validated YAML
configuration and then runs the shared staging, training, and save workflow.
"""
