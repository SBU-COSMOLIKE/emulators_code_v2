# The emulator/ package: code map

How to run the drivers and how to configure the YAML live in
[`../README.md`](../README.md); this file maps the package internals — how
it is laid out, what each file does, and where to edit for a given change.

## Contents

1. [Layout](#1-layout)
2. [The five emulator families](#2-the-five-emulator-families)
3. [What each file does](#3-what-each-file-does)
4. [Change X → edit Y](#4-change-x--edit-y)
5. [Variants](#5-variants)
6. [Every file's functions](#6-every-files-functions)

---

## 1. Layout

```
emulator/                              the library (torch; cosmolike only in geometries/output)
  data_staging.py                      load dumps -> "source" dicts; the physical cut
  geometries/                          the geometry family folder:
    parameter.py                       INPUT whitening (params -> network input)
    output.py                          the 3x2pt data-vector geometry (imports cosmolike)
    scalar.py                          derived-parameter standardization
    cmb.py                             the per-multipole diagonal C_ell geometry
    grid.py                            background functions of z + the log_offset law
    grid2d.py                          the (z, k) matter-power surfaces + the syren laws
  background.py                        H(z) -> distances: the imposed background physics
  syren_base.py                        the syren analytic P(k) base the MPS emulators correct
  analytics.py                         analytic xi rescaling R (optional preprocessing)
  activations.py                       learnable activations (H + variants)
  designs/                             the model family (section 5):
    blocks.py                          Affine, ResBlock, BinLinear, TRFBlock, FiLMGenerator
    plain.py                           ResMLP, ResCNN, ResTRF
    ia.py                              factored-IA templates (TemplateMLP, ...)
    pce.py                             PCEEmulator: the closed-form NPCE base
  losses/                              the chi2 family (sections 2 + 5):
    core.py                            chi2 losses + make_chi2 + anneal_value
    ia.py                              factored-IA loss + amplitude coefficients
    pce.py                             NPCE losses (residual / ratio)
    scalar.py                          ScalarChi2 (also serves the grid + grid2d families)
    cmb.py                             the CMB chi2 pair + the amplitude-law registry
                                       + the residual-roughness term
    transfer.py                        the frozen-base transfer chi2
  batching.py                          memory sizing + regime-aware data loaders
  training.py                          build model/opt/sched, training loop, run_emulator
  experiment.py                        EmulatorExperiment: the whole setup as one object
  warmstart.py                         fine-tune / transfer sources: load, extend, transfer
  scheduling.py                        GPU job balancing + worker pool + VRAM packing
  studies/                             Optuna study identity and journal contracts:
    implementation.py                 semantic implementation-version registry
    manifest.py                       scientific identity records + journal binding
    manifest_digest.py                canonical JSON + SHA-256 helpers
    name.py                           stable family-owned study names
  results.py                           save_learning_curves; save_emulator + rebuild_emulator
  inference.py                         EmulatorPredictor: rebuild + predict (every family)
  plotting.py                          history / curves / coverage / the diagnostics PDF
  diagnostics.py                       coverage, floor, hard directions + per-family pages
  family_drivers.py                    the shared sweep-block helpers (the
                                       family tune/sweep drivers and the
                                       CMB/BAOSN/MPS trainers are wrappers;
                                       scalar_train is standalone — no dv)
  cocoa.py                             the cocoa project layout (paths, YAML resolution)

*_train_emulator.py                    CLI: one training run (+ optional diagnostics
                                       PDF); cosmic_shear_ is the engine that the
                                       family-pinned wrappers ride
*_tune_emulator.py                     CLI: Optuna search (multi-GPU journal study
                                       for every family; per-family study names)
*_sweep_ntrain_emulator.py             CLI: f(dchi2 > thr) vs N_train (same split)
*_sweep_hyperparam_emulator.py         CLI: sweep ONE YAML-chosen knob
                                       (multi-GPU + --gpu-pack, every family)
cosmic_shear_bakeoff_activation_*.py   CLI: one curve per activation    (multi-GPU)
example_yamls/                         template YAMLs; copy one into a project's --fileroot
cobaya_theory/                         one thin cobaya Theory adapter per artifact kind
syren/                                 the VENDORED syren (symbolic_pofk) P(k) formulas
                                       the MPS emulators correct (numpy-only; provenance
                                       + the import-only deviations in syren/README.md)
compute_data_vectors/                  the training-set generators + the CMB covariance
ai/gates/                                 the acceptance board (see ai/gates/README.md; board.py is the authoritative gate registry)
```

The driver scripts sit beside `emulator/` (no `driver/` subfolder): launching one
puts its own folder on `sys.path`, so `import emulator` resolves with no path
setup. In a CoCoA install this folder is
`external_modules/code/emulators_code_v2/`; run the drivers from `$ROOTDIR`.
The `cobaya_theory/` adapters drive `emulator/inference.py`
(`EmulatorPredictor`) to run saved emulators inside a Cobaya MCMC (each
prepends the repo root to `sys.path` so `import emulator` resolves from a
folder deeper).

Only cosmic-shear training imports CosmoLike, through
`geometries/output.py`. The other training families avoid CosmoLike but still
use PyTorch, NumPy, SciPy, YAML, HDF5, and psutil. Plotting and diagnostics
have optional Matplotlib and GetDist dependencies. Data generation and the
integration smokes can additionally require Cobaya and CAMB.

---

## 2. The five emulator families

One training stack serves five output kinds. The config's data block selects
the family. Every loss returns one score per cosmology, with shape `(B,)`, so
the shared trimming, focus, reduction, and training mechanics can consume it.
The scientific meaning of that score is family-specific:

| family | data key | geometry | per-row score | Cobaya adapter |
|---|---|---|---|---|
| cosmic shear | CosmoLike keys | `geometries/output.py` | $r^{\mathsf T}\Sigma^{-1}r$ | `emul_cosmic_shear` |
| scalar | `outputs` | `geometries/scalar.py` | squared standardized output residuals | `emul_scalars` |
| CMB spectra | `cmb` | `geometries/cmb.py` | sum of `((prediction - truth) / sigma)^2`, using the stored per-ℓ `sigma` | `emul_cmb` |
| background | `grid` | `geometries/grid.py` | squared standardized residuals in encoded law space | `emul_baosn` |
| matter power | `grid2d` | `geometries/grid2d.py` | squared standardized residuals in encoded law space | `emul_mps` |

A `pce:` block first fits and freezes a finite Legendre-polynomial base, then
trains the selected neural architecture as a residual refiner. On CMB it
requires `amplitude_law: none`.

`background.py` owns the distance integrations used by the BAOSN adapter,
diagnostics, and gates. The background generator obtains its raw quantities
directly from CAMB. `syren_base.py` is shared by the optional MPS base-sidecar
producer, the MPS adapter, and gates.

---

## 3. What each file does

**Data & geometry**

| File | Role |
|---|---|
| `data_staging.py` | Builds a resident Python source dictionary whose arrays may be compact RAM copies or file-backed memmaps. Parameter text is parsed eagerly; target `.npy` files are memory-mapped. If the selected rows fit the memory budget, both arrays are copied into compact local order. Otherwise the target payload stays file-backed and `idx` keeps source-row coordinates. `dump_rows` preserves original row identity for aligned sibling files. The grid2d path thins columns before forming its law-space target in bounded chunks. |
| `geometries/parameter.py` | Input whitening: `ParamGeometry` (center + rotate into the covmat eigenbasis + unit-scale), `LogParamGeometry`, and the IA-factoring `AmplitudeFactorGeometry`. Known numerical gap (fix queued): covariance eigenvalues and rebuilt scales are not yet validated before division; details in `ai/notes/artifacts-inference-warmstart.md`. |
| `geometries/output.py` | The 3x2pt output side: `DataVectorGeometry` (squeeze to unmasked entries, whiten, own the chi2 `Cinv`), `DiagonalGeometry`, `BlockDiagonalGeometry`, `build_shear_angle_map`. **Only file importing cosmolike.** Known numerical gap (fix queued): a singular per-bin covariance is clipped to zero and then used as a divisor; details in `ai/notes/artifacts-inference-warmstart.md`. |
| `geometries/scalar.py` | `ScalarGeometry`: per-output standardization over named derived parameters, with the un-standardizable-column guard. |
| `geometries/cmb.py` | `CmbDiagonalGeometry`: subtract the training center and divide by the persisted `sigma_<spectrum>` from the covariance `.npz`; persist the spectrum, multipole grid, units, and amplitude-law facts. |
| `geometries/grid.py` | `GridGeometry` + `TARGET_LAWS`: a function on a stored z grid, the law (`log_offset` / `none`) inside encode/decode. |
| `geometries/grid2d.py` | `Grid2DGeometry` + `TARGET_LAWS_2D`: a flattened (z, k) surface standardized in LAW space (the syren division happens at staging; see `syren_base.py`). |
| `background.py` | The BAOSN imposed physics: the cumulative Simpson (composite even nodes + the correct one-interval odd node), c/H on the doubled grid, the flat distance conversions, `distance_interpolators`. |
| `syren_base.py` | The syren base surface the MPS emulators correct (`base_pklin`, `base_boost`) + `syren_params_from` (the ONE rule mapping resolved parameters to the base's arguments — generator and adapter cannot disagree). The formulas themselves are vendored in `syren/` (numpy-only), so the imports are unconditional. |
| `analytics.py` | Closed-form analytic xi (Eisenstein-Hu) to divide out broadband cosmology dependence — the optional rescaling `R`. |

**Model**

| File | Role |
|---|---|
| `activations.py` | Learnable activations: the paper's `H` plus Power / Gated / GatedPower variants; `make_activation` maps a name → factory. |
| `designs/blocks.py` | The small `nn.Module`s models are built from: `Affine`, `ResBlock`, `BinLinear`, `TRFBlock`, `FiLMGenerator`, plus `rescale_kernel_size`. |
| `designs/plain.py` | `ResMLP` is the trunk-only neural architecture. `ResCNN` and `ResTRF` add a coordinate-aware correction head to that trunk. Scalar outputs have names but no ordered output axis, so the scalar family accepts `ResMLP` only. The factored-IA and NPCE variants are in `designs/ia.py` and `designs/pce.py`. |

**Loss & training**

| File | Role |
|---|---|
| `losses/core.py` | chi2 losses on the whitened residual: `CosmolikeChi2` (plain; the `sqrt` / pseudo-Huber / `berhu` / `berhu_capped` mode ladder), `RescaledChi2` / `ResidualBaseChi2` (analytic-R), `ElementWeightedChi2`; `anneal_value`; `make_chi2`. The `_reduce` here is THE shared reduction every family's loss routes through. |
| `losses/scalar.py` | `ScalarChi2` + `make_scalar_chi2`: the standardized-residual chi2 (also wraps the grid and grid2d geometries — their laws live in the geometry, so the loss needs nothing new). |
| `losses/cmb.py` | `AMPLITUDE_LAWS` (`none` / `as_exp2tau_ref`), `CmbDiagonalChi2` / `CmbFactoredChi2` (the imposed-amplitude target, measured against the persisted fiducial so the factor is order-one), `ResidualRoughness` (the optional band-explicit penalty on short-period residual wiggles), `make_cmb_chi2`. |
| `losses/transfer.py` | `TransferChi2`: a frozen base network under a parallel correction, using gain or sum in physical or whitened space on the CosmoLike path. `TransferDiagChi2` applies the same design to CMB law-none, grid, and grid2d artifacts in whitened space only. |
| `batching.py` | Memory sizing + the regime-aware loaders (GPU-resident / RAM-stream / memmap-stream) that feed the training loop. Known gap (fix queued): validation computes its own safe chunk against the remaining budget, but the training loop uses the train chunk for validation and can exceed that bound; details in `ai/notes/training-stack.md`. |
| `training.py` | Device pick, the `make_model/optimizer/scheduler` factories, `build_run_specs`, the `[default, min, max, kind]` search resolvers, the config validators (`validate_phase_block` / `validate_loss` — now with the `roughness:` sub-block — / `validate_berhu` / `validate_ema`), the per-epoch loop, and `run_emulator`. |

**Orchestration & output**

| File | Role |
|---|---|
| `experiment.py` | `EmulatorExperiment`: config → device → data → geometry → chi2 → spec → train as one reusable object (`from_yaml` / `from_config`). Holds every family's validator (`validate_scalar` / `validate_cmb` / `validate_grid` / `validate_grid2d` / `validate_param_cuts` / `validate_sizes`), the family branches of `from_config` / `build_geometry` (including the fine-tune geometry pins), and the grid2d staging law transform. The drivers compose it. |
| `warmstart.py` | Fine-tune / transfer sources: `load_source` (validate a saved artifact), `extend_input_geometry` (block-extend for new parameters), `pin_output_geometry` (the cosmolike pin; the scalar/cmb/grid/grid2d pins live in their `build_geometry` branches), `build_warm_start` (transfer the weights + prove epoch-0 parity), `anchor_masks`. |
| `scheduling.py` | GPU job balancing (`lpt_assign`, `even_assign`), the spawned worker pool (`run_gpu_pool`), and the `--gpu-pack` VRAM-token machinery. Known lifecycle gap (fix queued): early failures do not yet guarantee every sibling process is terminated/joined, and invalid token plans can wait forever; details in `ai/notes/training-stack.md`. |
| `studies/implementation.py` | The semantic implementation-version registry included in every tuning-study identity. |
| `studies/manifest.py` | Builds complete scientific identity records and binds/verifies them on Optuna journal studies. |
| `studies/manifest_digest.py` | Strict canonical JSON plus stable value and file SHA-256 helpers. |
| `studies/name.py` | Resolves the stable family-owned Optuna study name independently of a wrapper filename. |
| `results.py` | `save_learning_curves` / `save_sweep_table`; `save_emulator` writes CPU-normalized tensors to `.emul` and the recipe and geometry record to `.h5`; `rebuild_emulator` reads the pair and uses `map_location=device` to choose the destination device. Its `info` dictionary carries the family flags and family-specific artifact facts. |
| `inference.py` | `EmulatorPredictor`: rebuild a saved emulator and predict — one `predict(params)` for every artifact kind: the dv section, the scalar `{name: value}` dict, the physical C_ell row, the background `{"z", quantity}` function, or the (z, k) law-space surface. Reuses the exact training decode per family. |
| `plotting.py` | Training history, learning-curve overlays, coverage panels, xi curves, and the multipage diagnostics PDF with the per-family pages (`cmb=` / `scalar=` / `grid=` / `grid2d=`). |
| `diagnostics.py` | Post-training analyses: the family-generic chi2 trio (coverage, local-linear floor, hard directions) + the per-family physical analyses (`cmb_residual_diagnostic`, `scalar_output_diagnostic`, `grid_residual_diagnostic`, `grid2d_residual_diagnostic`). Known production-width gap (fix queued): the local-linear floor materializes validation rows x 40 neighbours x output width, so an MPS `--diagnostic` run is not bounded; details in `ai/notes/training-stack.md`. |
| `family_drivers.py` | The shared sweep-block helpers (`read_sweep_block`, `set_by_path`, `SWEEPABLE_TOP_KEYS`, `ACTIVATION_PATHS`) — one definition of the YAML `sweep:` block, imported by the cosmic-shear one-knob driver. Every per-family tune/sweep driver is a thin wrapper over `main(prog, family)`, and the CMB/BAOSN/MPS trainers are wrappers too, so the multi-GPU pool, `--gpu-pack`, and the Optuna journal study carry over. `scalar_train_emulator.py` is standalone: its file naming and provenance have no data-vector file. |
| `cocoa.py` | The cocoa project layout: `--root` / `--fileroot` / `--yaml` resolution, output paths. |

**Drivers** (beside `emulator/`; each reads `--root` / `--fileroot` / `--yaml`)

| File | Role |
|---|---|
| `cosmic_shear_train_emulator.py` | One training run — cosmic shear, cmb, grid, or grid2d (the data block picks the family); `--diagnostic` writes the multipage PDF with that family's pages. |
| `scalar_train_emulator.py` | One scalar training run; `--diagnostic` adds the scalar pages. |
| `{cmb,baosn,mps}_train_emulator.py` | Thin family wrappers over the cosmic-shear driver's `main()`: each pins its data-block family (`cmb` / `grid` / `grid2d`), so a wrong-family YAML fails naming the right driver (`require_family_block`). |
| `cosmic_shear_tune_emulator.py` | Optuna study; multi-GPU via a shared journal-file study. Known accounting gap (fix queued): the parent does not inspect worker exit codes and can accept an old COMPLETE trial after current-worker failures; details in `ai/notes/training-stack.md`. |
| `{scalar,cmb,baosn,mps}_tune_emulator.py` | Thin wrappers over the cosmic-shear tune driver's `main(prog, family)`: the full capability (serial or `--n-gpus` journal study), the family pinned, per-family study names. |
| `cosmic_shear_sweep_ntrain_emulator.py` | `f(dchi2 > thr)` vs `N_train`; multi-GPU, LPT-balanced; `--gpu-pack`. |
| `{scalar,cmb,baosn,mps}_sweep_ntrain_emulator.py` | Thin wrappers over the cosmic-shear sweep driver's `main(prog, family, out_default)`: multi-GPU + `--gpu-pack` carry over; wrong-family YAMLs name the right driver. |
| `cosmic_shear_sweep_hyperparam_emulator.py` | Sweep ONE hyperparameter chosen in the YAML `sweep:` block; multi-GPU. |
| `{scalar,cmb,baosn,mps}_sweep_hyperparam_emulator.py` | Thin wrappers over the cosmic-shear one-knob driver's `main(prog, family, out_default)`: the same `sweep:` block, multi-GPU + `--gpu-pack`. |
| `cosmic_shear_bakeoff_activation_emulator.py` | One learning curve per activation; multi-GPU. Known lifecycle gap (fix queued): a worker failure before its training loop emits no result while the parent waits on a fixed number of un-timed queue reads; details in `ai/notes/training-stack.md`. |

Known learning-curve gap (fix queued): the optional-cut families' wrappers reach
`pool_size`, which currently indexes `omegabh2_hi` even when `param_cuts` is
absent. Their no-cut examples can fail before training; the shared selection
contract is in `ai/notes/data-generation-and-cuts.md`.

The naming rule for every driver is `<family>_<verb>_emulator.py` — what
you are emulating comes first, always (the 2026-07-11 family-first
rename; the board configs moved with it).

**cobaya_theory/** (one thin adapter per artifact kind; each rejects the
other kinds' artifacts loudly, naming the right adapter)

| File | Serves |
|---|---|
| `emul_cosmic_shear.py` | data-vector artifacts → the likelihood's dv (`state["cosmic_shear"]`). |
| `emul_scalars.py` | scalar artifacts → named derived parameters (provides read FROM the artifacts). Known Cobaya-contract gap (fix queued): `calculate` does not create `state["derived"]`; the stubbed identity gate never calls that path. |
| `emul_cmb.py` | CMB artifacts → the cobaya Cl dict (spectra / lmax / units are artifact facts; `must_provide` refuses beyond-training requests). |
| `emul_baosn.py` | the H(z) + recombination-D_M pair → Hubble + distances, served PIECEWISE by redshift window (the desert between the windows is a loud error, never a bridge); flat-only V1. |
| `emul_mps.py` | the pklin + boost pair → `get_Pk_grid` / `get_Pk_interpolator` (linear + nonlinear) for cosmolike's hybrid mode (`use_emulator: 2`); multiplies the syren base back per the artifacts' stored laws. Known defects (fix in progress): the derived-sigma8 helper uses 8 Mpc against k in 1/Mpc, its products are advertised through Cobaya's input-support hook, and its Syren requirement narrows the shared `As_1e9`/`As` rule to `As`; do not treat the stubbed identity gate as a real Cobaya dependency test (ai/notes/artifacts-inference-warmstart.md). |

**compute_data_vectors/** (the training-set generators)

| File | Role |
|---|---|
| `generator_core.py` | The shared machinery: the CLI (identical flags for every driver), emcee/uniform sampling, the chain + `.paramnames` + `.ranges` + `.covmat` writers, checkpoint save/load/append, the RAM-aware dv store, the MPI master/worker farm. Drivers subclass `GeneratorCore` and override only the probe whitelist, their train_args keys, the dv store hooks, and `_compute_dvs_from_sample`. Known gaps (fix in progress): failed samples stay as zeroed dv rows while staging ignores the failfile; and checkpoint append is not one manifest-bound transaction — omitted sidecars or a load error can fall through to fresh generation on the same roots. |
| `dataset_generator_lensing.py` | cosmolike data vectors (cs / ggl / gc); the core's default single-2D store. |
| `dataset_generator_cmb.py` | CMB spectra: four per-spectrum 2D files (tt / te / ee / pp) from one CAMB pass, phi-phi filled. |
| `dataset_generator_background.py` | H(z) on the SN grid + D_M on the recombination window, one background-only CAMB evaluation per sample, grid sidecars beside the dumps. |
| `dataset_generator_mps.py` | linear P + boost on the (z, k) grids (+ the syren base files when `write_syren_base`), through the Pk_interpolator requirement (the wants-Cl quirk kept verbatim). |
| `compute_cmb_covariance.py` | Computes the Gaussian CMB covariance and, when requested, dense lens-induced blocks with a five-point-stencil convergence study. The `.npz` stores the multipole grid, per-spectrum `sigma`, fiducial spectra, optional dense blocks, and provenance. Current training consumes the grid, one `sigma`, and one fiducial spectrum; it does not consume the dense blocks. |

---

## 4. Change X → edit Y

| To change… | Edit |
|---|---|
| a model architecture | `designs/plain.py` (+ `designs/blocks.py`); the IA / NPCE variants in `designs/ia.py` / `designs/pce.py`; register in `experiment.py`'s `models` |
| an activation function | `activations.py` (register it in `make_activation`) |
| the loss / add a chi2 variant | `losses/core.py` (variants in their family files); the shared reduction is `_reduce` there |
| a new imposed target law | the family geometry's registry + its executor (`losses/cmb.py` for amplitude laws; the geometry itself for grid laws; `syren_base.py` for a cosmology-dependent base) |
| update the syren base formulas | re-vendor `syren/` deliberately (see `syren/README.md`) and RETRAIN the MPS artifacts — the base dumps beside old training data still carry the formula they were generated with |
| how parameters are whitened (input) | `geometries/parameter.py` |
| dv whitening / cosmolike reading (output) | `geometries/output.py` |
| a NEW output family | a geometry module (+ `state`/`from_state`), a loss (or reuse `ScalarChi2`), a `validate_<family>` + `from_config` branch + `build_geometry` branch in `experiment.py`, a predictor branch in `inference.py`, an adapter in `cobaya_theory/`, two gates — scalar / cmb / grid / grid2d are four worked examples |
| data loading / the physical cut / staging | `data_staging.py` |
| restrict the training pool by a density window | the `data.param_cuts:` sub-block; a new window is one row in `data_staging.phys_cut_idx`'s table |
| the GPU-memory regime / batching | `batching.py` |
| the optimizer/scheduler build or the training loop | `training.py` |
| the end-to-end setup wiring | `experiment.py` |
| fine-tuning / transfer mechanics | `warmstart.py` (+ the family pin in `experiment.build_geometry`) |
| what an artifact stores | `results.py` save + rebuild TOGETHER (+ the geometry's `state()`) |
| how a saved emulator is served | the predictor branch in `inference.py`, then the thin adapter in `cobaya_theory/` |
| a CLI driver (add/modify) | the `<family>_<verb>_emulator.py` beside `emulator/` (tune/sweep family versions and the CMB/BAOSN/MPS trainers are thin wrappers over the cosmic-shear mains; scalar train is the standalone, data-vector-free exception) |
| which hyperparameters are searched | the driver YAML (`[default, min, max, kind]`) + resolvers in `training.py` |
| tuning-study identity, journal authentication, or stable family name | the matching owner in `studies/` (`manifest.py`, `manifest_digest.py`, `implementation.py`, or `name.py`) |
| multi-GPU balancing | `scheduling.py` |
| the training-set sampling / checkpoints / MPI farm | `compute_data_vectors/generator_core.py` (all four generators inherit) |
| one generator's physics | that generator's `_compute_dvs_from_sample` only |
| the output file format | `results.py` |
| a plot | `plotting.py` |
| a diagnostic | `diagnostics.py` |
| analytic preprocessing | `analytics.py` |

---

## 5. Variants

The design/loss family folders each carry the main path plus its variants,
one file per variant. Each pairs a design (or a frozen base) with its own
loss.

| Variant | Design + loss | What it is |
|---|---|---|
| NPCE | `designs/pce.py` + `losses/pce.py` | a sparse-Legendre polynomial-chaos base plus a neural refiner. |
| Factored IA | `designs/ia.py` + `losses/ia.py` | emulate cosmology-only templates and apply the IA-amplitude polynomial in closed form (the amplitudes never enter the network). |
| Transfer | `warmstart.py` + `losses/transfer.py` | a frozen trained base under a parallel correction network. Supported for CosmoLike, CMB, grid, and grid2d artifacts; scalar artifacts use fine-tuning instead. |
| Fine-tuning | `warmstart.py` (`train_args.finetune`) | warm-start from a compatible saved source. The copied parameters are exact; the rebuilt function must agree within the parity tolerance, and zero-connected new inputs must have bit-identical zero influence. Supported by every family. |

The grid2d NPCE fit materializes the thinned target on the accelerator and
runs a dense float64 singular-value decomposition on the CPU. Use it only for
research-scale fits that fit both memory budgets.

---

## 6. Every file's functions

One line per function / class / method. For full detail, read the docstring in
the file itself; this is the index.

### `emulator/data_staging.py` <a name="apx-data_staging"></a>

Turns on-disk dumps into in-memory "source" dicts.

- `load_source(...)` — orchestrator: memmap the dv, load + cut the params, keep `N_train` rows, stage, return `{C, dv, idx, dump_rows (+ means)}` (`dump_rows` = the staged rows' on-disk indices, for sibling-file alignment).
- `load_scalar_source(...)` — the scalar sibling: inputs AND outputs are named columns of one parameter `.txt` (the getdist `.paramnames` sidecar locates them).
- `stage_source(C, dv, idx, ram_frac)` — materialize the used rows in RAM if they fit, else keep the memmap (reindex local).
- `phys_cut_idx(...)` — keep the rows inside every active physical-density window; returns `(kept_idx, report)`.
- `stream_chunks` / `stream_stats` / `param_stats` — streamed per-column stats (the dump never loads whole).
- `read_param_names(covmat_path, comment)` — parameter names from the covmat header line.
- `check_paramnames` / `_scalar_columns` — the sidecar-vs-covmat naming integrity checks.

### `emulator/geometries/parameter.py` <a name="apx-geometries_parameter"></a>

Input side: raw parameters → whitened network input.

- `ParamGeometry` — center, rotate into the parameter-covariance eigenbasis, unit-scale (`from_covmat` / `from_state` / `state`; `whiten`/`unwhiten`, `encode`/`decode`).
- `LogParamGeometry` — whitens in log space for multiplicative params.
- `AmplitudeFactorGeometry` — whiten every parameter except the IA amplitude(s), append them raw for the loss's closed-form combine.

### `emulator/geometries/output.py` <a name="apx-geometries_output"></a>

The 3x2pt output side: raw dv ↔ whitened masked target; holds the chi2
covariance. The only file importing cosmolike.

- `DataVectorGeometry` — the base geometry for one probe (`from_cosmolike` / `from_state` / `state`; `squeeze`/`unsqueeze`; `whiten`/`unwhiten`, `encode`/`decode`).
- `DiagonalGeometry` — whiten by the marginal sigma only (theta order kept, for a CNN).
- `BlockDiagonalGeometry` — whiten each tomographic bin by its own sub-block.
- `build_shear_angle_map(geom, ...)` — attach per-element theta / source-z / xi± branch / per-bin sizes.

### `emulator/geometries/scalar.py` <a name="apx-geometries_scalar"></a>

- `ScalarGeometry` — per-output standardization over named derived parameters (`from_targets` with the un-standardizable guard; `from_state` / `state`; `encode`/`decode`; `dest_idx`/`total_size` the trivial identity).

### `emulator/geometries/cmb.py` <a name="apx-geometries_cmb"></a>

- `CmbDiagonalGeometry` — subtracts the target center and divides by the persisted `sigma_<spectrum>` from the covariance `.npz`; persists the spectrum, multipole grid, units, and amplitude-law facts. `from_fiducial` is a synthetic-construction helper; the training path passes the stored scale through `__init__`.
- `attach_head_coords()` — the conv/TRF heads' channel/token split: one bin covering the spectrum, coordinate = ell; pure, idempotent, run at training and at rebuild.

### `emulator/geometries/grid.py` <a name="apx-geometries_grid"></a>

- `TARGET_LAWS` — `{none, log_offset}`; the law lives INSIDE encode/decode here.
- `GridGeometry` — a function on a stored z grid (`from_targets` applies the law first; persists quantity / units / law / offset / z).
- `attach_head_coords()` — the heads' split: one bin, coordinate = z.

### `emulator/geometries/grid2d.py` <a name="apx-geometries_grid2d"></a>

- `TARGET_LAWS_2D` — `{none, syren_linear, syren_halofit}` (names only: the cosmology-dependent base is the consumer's multiply, through `syren_base.py`).
- `Grid2DGeometry` stores a function on a two-dimensional grid. Redshift is the
  outer index and wavenumber is the inner index. The geometry standardizes the
  flattened values after applying the selected target law. Its saved state
  records the quantity, units, law and both axes. It also always records a
  `const_mask`. A false entry leaves that network output active. A true entry
  marks a value that is constant across every training cosmology, such as the
  low-wavenumber $B=1$ tail of a boost. Decode returns the stored training
  constant at those entries. An all-false mask explicitly records that no
  entries are pinned. A surface that is constant everywhere remains an error
  because it is the signature of a dead data dump.
- `attach_head_coords()` — the heads' split: one bin PER Z SLICE, length nk (conv channels / TRF tokens = z slices).

### `emulator/background.py` <a name="apx-background"></a>

The BAOSN imposed physics (one definition for the adapter AND direct scripts).

- `cumulative_simpson(z, y)` — even doubled-grid points exact on cubics; each odd node is the correct one-interval integral `h/12*(5,8,-1)`, exact on quadratics (superseding the earlier half-chunk approximation; see `ai/notes/families-background-mps.md`).
- `comoving_distance_grid(z_grid, h_grid)` — c/H cubic onto the doubled grid, Simpson.
- `distance_interpolators(z_grid, h_grid)` — the H / chi / D_A / D_L cubics + the window edge.

### `emulator/syren_base.py` <a name="apx-syren_base"></a>

The syren analytic P(k) base (one definition for the generator, `emul_mps`,
and the gates). The formulas import from the vendored `syren/` package
(numpy-only; provenance + deviations in `syren/README.md`).

- `syren_params_from(params)` — the ONE mapping rule from resolved parameters to the base arguments (`As`/`As_1e9`; an absent equation of state means LCDM on both sides).
- `base_pklin(...)` / `base_boost(...)` — the legacy `_compute_mps_approximation` / `_compute_boost_approximation` math verbatim (unit conventions included).

### `emulator/analytics.py` <a name="apx-analytics"></a>

Analytic xi rescaling `R` (Eisenstein-Hu zero-baryon preprocessor).

- `_analytic_R(...)` / `analytic_shape_ratio(...)` / `rescale_xi(...)`.

### `emulator/activations.py` <a name="apx-activations"></a>

Learnable activations for the ResBlock `act` slot.

- `activation_fcn` — the paper's `H` (a learnable identity↔Swish interpolation).
- `GatedActivation` / `PowerGatedActivation` / `GatedPowerActivation` — generalizations.
- `make_activation(name, n_gates)` — map a name to a factory `act(dim) -> module`.

### `emulator/designs/` (the model family) <a name="apx-designs"></a>

Shared `blocks.py`, the plain models in `plain.py`, the factored-IA / NPCE
variants in `ia.py` / `pce.py`. The class flags the config layer reads:
`head_block` (which correction head, None = trunk-only), `factored` (IA
amplitudes appended raw), `needs_bins` (wants the shear-angle map).

- `designs/blocks.py` — `Affine`, `FeatureAffine`, `make_norm`, `ResBlock`, `BinLinear`, `TRFBlock`, `FiLMGenerator`, `rescale_kernel_size`.
- `designs/plain.py` — `ResMLP` (input projection → residual blocks → output projection → Affine), `ResCNN` (gated bins-as-channels 1D CNN correction in theta order), `ResTRF` (bin-token transformer correction). The head knobs live under YAML `model.cnn` / `model.trf`; see each class docstring for the full knob table and the shape-flow diagrams.
- `designs/ia.py` — `TemplateMLP`, `TemplateResCNN`, `TemplateResTRF` (cosmology-only templates; the amplitudes never enter the network).
- `designs/pce.py` — `PCEEmulator` (the closed-form sparse-Legendre base; loss-owned, not an SGD architecture) + `pce_multi_index`, `pce_design`, `select_lars_loo`.

### `emulator/losses/` (the chi2 family) <a name="apx-losses"></a>

Each loss holds a geometry (composition) and routes through `core.py`'s
shared `_reduce`.

- `losses/core.py` — `anneal_value`; `CosmolikeChi2` (the plain chi2 + the `chi2`/`sqrt`/`sqrt_dchi2`/`berhu`/`berhu_capped` transform ladder); `RescaledChi2` / `ResidualBaseChi2` (analytic-R); `ElementWeightedChi2`; `make_chi2`.
- `losses/ia.py` — `nla_coeffs`, `tatt_coeffs`; `NLAAmpFactoredChi2`, `TemplateFactoredChi2`.
- `losses/pce.py` — `PCEResidualChi2`, `PCERatioChi2` (the cosmolike forms); `PCEResidualDiagChi2` (the family-wide residual form over the elementwise-whitened geometries — cmb law-none / grid / grid2d / scalar; roughness composes).
- `losses/scalar.py` — `ScalarChi2` + `make_scalar_chi2` (the standardized-residual chi2; also wraps `GridGeometry` and `Grid2DGeometry` — their laws live in the geometry).
- `losses/cmb.py` — `AMPLITUDE_LAWS`; `CmbDiagonalChi2` (plain per-multipole chi2) / `CmbFactoredChi2` (the imposed `C_ell e^{2tau}/A_s` target, reading named columns); `ResidualRoughness` + `configure_roughness` (the optional short-period residual penalty, byte-identical when absent); `make_cmb_chi2`.
- `losses/transfer.py` — `TransferChi2` (a frozen base + a parallel correction; gain/sum, physical/whitened; cosmolike); `TransferDiagChi2` (the diagonal-family form: whitened space only, plain bases, refine/roughness refused loudly).

### `emulator/batching.py` <a name="apx-batching"></a>

Memory sizing and the regime-aware data loaders (GPU-resident / RAM-stream /
memmap-stream).

- `compute_batch_size_bytes` / `compute_model_size_bytes` / `batches_per_load` — memory estimates.
- `_build_loaders_one(...)` / `build_loaders(...)` — pick a regime per source; return the loaders the loop consumes.

### `emulator/training.py` <a name="apx-training"></a>

The run layer that ties everything together.

- `pick_device` / `make_logger` — setup helpers.
- `make_model` / `make_optimizer` / `make_scheduler` — build one component from a `{cls, **kwargs}` spec dict.
- `build_run_specs(...)` — config → the `run_emulator` spec dicts (model / optimizer / lr / scheduler / trim / focus are REQUIRED blocks: plain subscripts, no code defaults).
- `validate_phase_block` / `validate_loss` / `validate_berhu` / `validate_ema` — the pure config validators; `validate_loss` resolves `{mode, berhu, roughness}` (the roughness sub-block is the CMB residual-roughness term, top-level loss only).
- `derive_eval_bs` / `derive_ema_beta` — derived run values.
- `default_train_args` / `suggest_train_args` / `search_defaults` — the `[default, min, max, kind]` search resolvers.
- `eval_val` / `eval_source_chi2` — score the model on the val set / per-cosmology delta-chi2.
- `training_loop_batched(...)` — the per-epoch loop (trim / focus / berhu-blend / EMA annealing, best-epoch tracking, the optional Polyak average).
- `run_emulator(...)` — top-level: build model + optimizer + scheduler + loaders, apply the roughness term to a CMB chi2fn, train, return the histories.
- `audit_devices(model, lossfn, device)` — tensor-placement check.

### `emulator/experiment.py` <a name="apx-experiment"></a>

`EmulatorExperiment`: the whole setup as one reusable object.

- `from_yaml` / `from_config` — build from a YAML file / a parsed dict; `from_config` dispatches the FAMILY (scalar / cmb / grid / grid2d branches, each with its fine-tune sub-path) before the cosmic-shear path.
- `validate_param_cuts` / `validate_sizes` / `validate_scalar` / `validate_cmb` / `validate_grid` / `validate_grid2d` / `validate_transfer` — the pure data-block validators, one per concern.
- `resolve_phase_args` / `validate_sweep_paths` — the two-phase schedule resolution.
- `_head_activation_spec` / `_resolve_head_activation` / `_activation_flag_notice` / `_pinned_head_warning` — the per-head activation config layer.
- `stage_train` / `stage_val` / `pool_size` — stage the sources and report the physical-cut pool size. The grid2d branch thins the k axis, forms law-space rows in bounded chunks, computes moments from the stored float32 payload, and uses an experiment-owned temporary memmap when the resident copy would exceed its memory budget.
- `build_geometry` / `build_specs` — the input/output geometry + chi2 per family (the fine-tune pins live here); the `run_emulator` spec dicts.
- `train` / `run` / `frac_above` — train on the staged data; the full pipeline; the sweep metric.
- `print_design()` — the shared startup banner (family line included), so a stale YAML is caught at launch.

### `emulator/warmstart.py` <a name="apx-warmstart"></a>

Fine-tune / transfer sources (the warm-start machinery every family's finetune
routes through).

- `validate_finetune_config(...)` — the config-time whitelist (architecture / activation / loss form inherited; no model: block).
- `resolve_source_root` / `load_source` — resolve + validate a saved source artifact (plain input geometry, rescale none, no chaining).
- `recipe_to_model_opts` / `extend_input_geometry` / `pin_output_geometry` — the recipe → live specs, the block-extension for new parameters, the cosmolike output pin (the other families pin in their `build_geometry` branches).
- `build_warm_start(...)` — transfer the weights (zero-padded extra columns) + prove epoch-0 parity; `transfer_state_dict`, `anchor_masks`, `build_transfer_start`, `_zero_final_linear`.

### `emulator/scheduling.py` <a name="apx-scheduling"></a>

- `lpt_assign` / `even_assign` — GPU job splits.
- `run_gpu_pool(...)` — the spawned worker pool (one process per GPU lane).
- `estimate_train_vram_fraction` / `vram_tokens` — the `--gpu-pack` machinery.

### `emulator/studies/` (Optuna identity) <a name="apx-studies"></a>

- `studies/implementation.py` — `study_implementation_identity`, backed by
  the versioned registry of scientific tuner components.
- `studies/manifest.py` — `build_study_manifest`, `bind_study_manifest`, and
  `require_worker_identity`; the complete scientific identity and journal
  authentication boundary.
- `studies/manifest_digest.py` — `canonical_json`, `manifest_digest`, and
  `file_digest`; strict canonical bytes and SHA-256 identities.
- `studies/name.py` — `resolve_study_name`; stable names for all five emulator
  families, independent of wrapper filenames.

### `emulator/results.py` <a name="apx-results"></a>

- `save_learning_curves` / `save_sweep_table` — plain-text tables.
- `save_emulator(...)` — writes a CPU-normalized PyTorch `state_dict` to `.emul` and geometry states, histories, resolved config, model recipe, and optional PCE or transfer-base records to `.h5`.
- `rebuild_emulator(path_root, device)` — reconstructs `(model, param_geometry, geometry, info)` from the pair. `map_location=device` selects the destination, so loading does not require the accelerator used during saving. `info` carries the family flags and family-specific artifact facts.

### `emulator/inference.py` <a name="apx-inference"></a>

- `EmulatorPredictor` — rebuild + predict for every artifact kind. `__init__` branches on the rebuilt family (scalar → output names; cmb → spectrum/ell/units + the law-dispatched decoder; grid → quantity/z; grid2d → quantity/z/k; else the dv path with the ia/pce/transfer decoder). Every family branch builds its decoder at init: `_build_diag_decoder` composes an NPCE base when the artifact carries a `pce` group (plain `geom.decode` otherwise, byte-identical). `predict(params)` returns the family's natural object (see the class docstring's table); `_build_decoder` / `_build_cmb_decoder` / `_build_diag_decoder` reuse the exact training `chi2fn.decode`.

### `emulator/plotting.py` <a name="apx-plotting"></a>

Figures (colorblind-safe palette, no red/green).

- `plot_history` / `plot_learning_curves` / `plot_sweep_curve` / `plot_diagnostics` — the public figures; `plot_diagnostics` appends the per-family pages when given `cmb=` / `scalar=` / `grid=` / `grid2d=` (absent = the cosmic-shear PDF, byte-identical).
- `plot_xi` / `dv_to_xi` / `source_param_samples` — xi curves and the coverage triangle.
- `_history_panels` / `_coverage_panels` / `_floor_panel` / `_hard_direction_panels` / `_lcdm_triangle_fig` / `_lnparam_pca_fig` / `_cmb_pages` / `_scalar_pages` / `_grid_pages` / `_finish` / `_save_pages` — the panel and page builders.

### `emulator/diagnostics.py` <a name="apx-diagnostics"></a>

Post-training analyses (each returns a dict the plotting reads).

- `coverage_diagnostic(...)` / `local_linear_floor(...)` / `hard_direction_regression(...)` — the family-generic chi2 trio (they consume only params + per-sample chi2).
- `cmb_residual_diagnostic(...)` — per-multipole residual bands (fractional AND in error-bar units), the worst-cosmology overlay, the high-pass wiggle content the roughness term targets.
- `scalar_output_diagnostic(...)` — per-output truth/prediction/residual tables (physical + standardized) + the bias-hunt inputs.
- `grid_residual_diagnostic(...)` — per-redshift residual bands + (for a Hubble artifact) the derived D_A / D_L bands through the REAL `background.py` pipeline.
- `grid2d_residual_diagnostic(...)` — the matter-power (z, k) residual surfaces in LAW space (under a syren law the residual = ln(P_pred / P_truth), the base cancels): median-|residual| + worst cosmology + per-k bands at three redshifts.

### `emulator/family_drivers.py` <a name="apx-family_drivers"></a>

- `set_by_path(train_args, path, value)` / `read_sweep_block(cfg)` + `SWEEPABLE_TOP_KEYS` / `ACTIVATION_PATHS` — the one definition of the `sweep:` block, imported by the cosmic-shear one-knob driver (the family drivers are wrappers over `main(prog, family)`, so they inherit it).

### drivers (beside `emulator/`) <a name="apx-drivers"></a>

Each `main()` reads `--root` / `--fileroot` / `--yaml`.

- `cosmic_shear_train_emulator.py` — one training run (any dv-shaped family) + the diagnostics PDF.
- `scalar_train_emulator.py` — one scalar run + the diagnostics PDF.
- `cmb_train_emulator.py` / `baosn_train_emulator.py` / `mps_train_emulator.py` — the thin family wrappers (`main(prog, family)` + `require_family_block`).
- `cosmic_shear_tune_emulator.py` — the Optuna driver (serial or the multi-GPU journal study); `{scalar,cmb,baosn,mps}_tune_emulator.py` — thin family wrappers over its `main(prog, family)`.
- `cosmic_shear_sweep_ntrain_emulator.py` — the learning-curve driver (multi-GPU, LPT, `--gpu-pack`); `{scalar,cmb,baosn,mps}_sweep_ntrain_emulator.py` — thin family wrappers.
- `cosmic_shear_sweep_hyperparam_emulator.py` — one YAML-chosen knob (multi-GPU); `{scalar,cmb,baosn,mps}_sweep_hyperparam_emulator.py` — thin family wrappers.
- `cosmic_shear_bakeoff_activation_emulator.py` — one curve per activation.
