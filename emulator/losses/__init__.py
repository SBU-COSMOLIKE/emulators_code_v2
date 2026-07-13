"""The emulator loss family: every chi2 variant, one file each.

The companion of the emulator/designs/ family. Each loss holds a
DataVectorGeometry (composition) and turns a model's whitened output
into the training chi2. Which variant lives where:

  core.py   the plain losses and the shared schedule (make_chi2,
            CosmolikeChi2, RescaledChi2, ResidualBaseChi2,
            ElementWeightedChi2, the sqrt / pseudo-Huber / berhu ladder,
            anneal_value). Every other loss subclasses CosmolikeChi2.
  ia.py     the factored intrinsic-alignment loss (TemplateFactoredChi2)
            and the amplitude-polynomial coefficients (nla_coeffs,
            tatt_coeffs) that combine the templates from designs/ia.py.
  pce.py    the NPCE losses (PCEResidualChi2, PCERatioChi2), a frozen
            PCEEmulator base under a neural refiner.

This __init__ is a map only: import the variant module you need (from
emulator.losses.core import make_chi2), not this package.
"""
