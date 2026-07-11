"""The vendored syren (symbolic_pofk) analytic P(k) formulas.

These are the exact analytic matter-power-spectrum formulas the MPS
emulators CORRECT: the training target is log(P / P_base) with P_base
computed from this package (through emulator/syren_base.py, the base's
only consumer-facing surface). Vendoring the source in-repo pins the
formula: the emulator artifacts and the base they were trained against
can never drift apart under a pip upgrade, and no external install is
needed at generation, training, or inference time.

Two modules, copied from the symbolic_pofk bundle shipped with the
legacy emulmps code (Deaglan Bartlett et al., MIT license — see
syren/LICENSE and syren/README.md for provenance and the import-only
deviations):

  linear.py       plin_emulated (the linear P(k) fit, w0waCDM),
                  get_approximate_D / growth_correction_R (approximate
                  growth), As_to_sigma8 — arXiv:2311.15865 +
                  arXiv:2410.14623.
  syrenhalofit.py run_halofit_vec (the syren-halofit nonlinear boost)
                  — arXiv:2402.17492.

Import the submodule you need (import syren.linear); this __init__
deliberately re-exports nothing, so the import census stays exact.
Both modules need numpy only.
"""
