"""Loss and score owners for each emulator output family.

Every loss object provides the training loop with a common protocol:
``encode``, ``decode``, one score per batch row, and a scalar ``loss``
reduction. The physical meaning of that score depends on the geometry. A
masked cosmic-shear vector uses a dense inverse covariance, while scalar and
grid families use standardized residuals and CMB uses stored multipole
scales. The modules are:

  core.py   the plain losses and the shared schedule (make_chi2,
            CosmolikeChi2, RescaledChi2, ResidualBaseChi2,
            ElementWeightedChi2, the sqrt / pseudo-Huber / berhu ladder,
            anneal_value) for masked covariance data vectors.
  ia.py     the factored intrinsic-alignment loss (TemplateFactoredChi2)
            and the amplitude-polynomial coefficients (nla_coeffs,
            tatt_coeffs) that combine the templates from designs/ia.py.
  pce.py    the NPCE losses (PCEResidualChi2, PCERatioChi2), a frozen
            PCEEmulator base under a neural refiner.
  scalar.py standardized scores used by scalar, grid, and grid2d outputs.
  cmb.py    diagonal CMB scores and the amplitude-law registry.
  transfer.py frozen-base plus correction composition for supported families.

This __init__ is a map only: import the variant module you need (from
emulator.losses.core import make_chi2), not this package.
"""
