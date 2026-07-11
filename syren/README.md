# syren/ — the vendored symbolic_pofk formulas

The analytic matter-power-spectrum formulas the MPS emulators correct
(the "syren" base): the network learns `log(P / P_base)` and the exact
formula is multiplied back at inference. This folder pins the formula
in-repo so the artifacts and their base can never drift apart under a
package upgrade, and so nothing needs `pip install symbolic_pofk`.

## Provenance

Copied 2026-07-11 from the symbolic_pofk bundle shipped with the
legacy emulmps code (`emulators_code/emulmps/emulmps_emul/
symbolic_pofk`) — the exact copy the legacy pipeline ran, including
its local edits — originally from
[DeaglanBartlett/symbolic_pofk](https://github.com/DeaglanBartlett/symbolic_pofk)
(MIT license, kept verbatim in [LICENSE](LICENSE)). Papers:
arXiv:2311.15865 (linear fit), arXiv:2402.17492 (syren-halofit),
arXiv:2410.14623 (w0waCDM extension).

## What is here (only the functions we use, plus their file-mates)

| file | the functions the pipeline calls |
|---|---|
| `linear.py` | `plin_emulated` (linear P(k), w0waCDM), `get_approximate_D`, `growth_correction_R`, `As_to_sigma8` |
| `syrenhalofit.py` | `run_halofit_vec` (the nonlinear boost, `return_boost=True`) |

The one repo consumer is `emulator/syren_base.py` (`base_pklin` /
`base_boost` / `syren_params_from`) — the dump generator, the
`emul_mps` adapter, and the gates all go through it, never through
this package directly.

## Deviations from the source files (import lines only)

Function bodies are byte-verbatim (AST-verified in the vendoring
probe). Three import-line deviations:

1. `linear.py`: `import warnings` dropped — the name is never used in
   the file.
2. `linear.py`: `import scipy.integrate` dropped — never used either;
   dropping it makes the package numpy-only, importable everywhere the
   emulator package is.
3. `syrenhalofit.py`: `import symbolic_pofk.linear as linear` →
   `import syren.linear as linear` (the internal import retargeted at
   this package).

Each file also carries a short provenance header comment. Nothing
else changed. If the upstream bundle is ever updated, re-vendor
deliberately and retrain — the artifacts record the base they were
trained against by construction (the generator writes the base dumps
beside the raw dumps).
