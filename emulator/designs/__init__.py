"""The emulator design family: every network variant, one file each.

Gathered here so the same-named files stop colliding across folders.
Which variant lives where:

  blocks.py   the shared nn building blocks the models are assembled
              from (Affine, ResBlock, BinLinear, TRFBlock, make_norm,
              FiLMGenerator, rescale_kernel_size).
  plain.py    the standard models (DesignSpec, ResMLP, ResCNN, ResTRF),
              mapping whitened parameters to the whitened data vector.
  ia.py       the factored intrinsic-alignment templates (TemplateMLP,
              TemplateResCNN, TemplateResTRF): the network emits only
              the cosmology-only templates, the loss combines them with
              the IA amplitudes in closed form.
  pce.py      the polynomial-chaos base (PCEEmulator), a closed-form
              sparse-Legendre expansion used as the NPCE base under a
              neural refiner.

The losses that pair with these live in the sibling emulator/losses/
family (core.py, ia.py, pce.py). This __init__ is a map only: import
the variant module you need (from emulator.designs.plain import ResMLP),
not this package.
"""
