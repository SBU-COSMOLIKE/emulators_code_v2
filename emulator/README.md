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
  geometries_*.py                      LEGACY SHIMS only: saved artifacts persist the OLD
                                       flat module paths, so these re-exports live forever;
                                       new code never imports them (a gate censuses it)
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
  results.py                           save_learning_curves; save_emulator + rebuild_emulator
  inference.py                         EmulatorPredictor: rebuild + predict (every family)
  plotting.py                          history / curves / coverage / the diagnostics PDF
  diagnostics.py                       coverage, floor, hard directions + per-family pages
  family_drivers.py                    the serial per-family sweep/tune loops
  cocoa.py                             the cocoa project layout (paths, YAML resolution)

train_single_*.py / train_scalar_*.py  CLI: one training run (+ optional diagnostics PDF)
tune_*.py                              CLI: Optuna search (cosmic shear multi-GPU; the
                                       per-family tune_{scalar,cmb,baosn,mps}_ serial)
sweep_ntrain_*.py                      CLI: f(dchi2 > thr) vs N_train (same split)
sweep_hyperparam_*.py                  CLI: sweep ONE YAML-chosen knob  (multi-GPU)
bakeoff_activation_*.py                CLI: one curve per activation    (multi-GPU)
example_yamls/                         template YAMLs; copy one into a project's --fileroot
cobaya_theory/                         one thin cobaya Theory adapter per artifact kind
compute_data_vectors/                  the training-set generators + the CMB covariance
gates/                                 the acceptance board (32 gates; see gates/README.md)
```

The driver scripts sit beside `emulator/` (no `driver/` subfolder): launching one
puts its own folder on `sys.path`, so `import emulator` resolves with no path
setup. In a cocoa install this folder is
`external_modules/code/emulators/emultrfv2/`; run the drivers from `$ROOTDIR`.
The `cobaya_theory/` adapters drive `emulator/inference.py`
(`EmulatorPredictor`) to run saved emulators inside a Cobaya MCMC (each
prepends the repo root to `sys.path` so `import emulator` resolves from a
folder deeper).

Only `geometries/output.py` imports cosmolike, so cosmic-shear training runs
on a machine with a working Cocoa installation; every other family (and the
library everywhere else) is pure PyTorch and reviewable anywhere.

---

## 2. The five emulator families

One training stack serves five output kinds. The family is picked by the
config's data block; every family's loss exposes the same per-sample chi2, so
trimming / the focal weight / the berhu ladder / EMA / fine-tuning compose
unchanged for all of them:

| family | data block key | output geometry | loss | cobaya adapter |
|---|---|---|---|---|
| cosmic shear (3x2pt) | the cosmolike keys | `geometries/output.py` | `losses/core.py` (+ ia / pce / transfer) | `emul_cosmic_shear` |
| scalar (derived params) | `outputs` | `geometries/scalar.py` | `losses/scalar.py` | `emul_scalars` |
| CMB spectra | `cmb` | `geometries/cmb.py` | `losses/cmb.py` | `emul_cmb` |
| background (BAO/SN) | `grid` | `geometries/grid.py` | `losses/scalar.py` (reused) | `emul_baosn` |
| matter power (EMUL2) | `grid2d` | `geometries/grid2d.py` | `losses/scalar.py` (reused) | `emul_mps` |

Two physics modules sit beside the geometries, each with exactly one
definition shared by its generator, its adapter, and its gates:
`background.py` (the distances are KNOWN physics integrated from the emulated
H(z)) and `syren_base.py` (the analytic formula the MPS emulators correct).

---

## 3. What each file does

**Data & geometry**

| File | Role |
|---|---|
| `data_staging.py` | On-disk dumps → in-memory "source" dicts; streaming per-column stats; the physical density windows (`omega_b h^2` bound, plus the optional `omegam^2 h^2` / `omegamh2` / `omegamh2·n_s` windows). Memmaps the dv dump (never loads it whole). Returns `dump_rows` so a sibling dump file (the MPS base dumps) can be row-aligned. |
| `geometries/parameter.py` | Input whitening: `ParamGeometry` (center + rotate into the covmat eigenbasis + unit-scale), `LogParamGeometry`, and the IA-factoring `AmplitudeFactorGeometry`. |
| `geometries/output.py` | The 3x2pt output side: `DataVectorGeometry` (squeeze to unmasked entries, whiten, own the chi2 `Cinv`), `DiagonalGeometry`, `BlockDiagonalGeometry`, `build_shear_angle_map`. **Only file importing cosmolike.** |
| `geometries/scalar.py` | `ScalarGeometry`: per-output standardization over named derived parameters, with the un-standardizable-column guard. |
| `geometries/cmb.py` | `CmbDiagonalGeometry`: per-multipole whitening by the cosmic-variance error bar (sigma from the covariance script's .npz), the spectrum / units / amplitude-law facts persisted. |
| `geometries/grid.py` | `GridGeometry` + `TARGET_LAWS`: a function on a stored z grid, the law (`log_offset` / `none`) inside encode/decode. |
| `geometries/grid2d.py` | `Grid2DGeometry` + `TARGET_LAWS_2D`: a flattened (z, k) surface standardized in LAW space (the syren division happens at staging; see `syren_base.py`). |
| `geometries_*.py` (flat) | The legacy shims — one re-export line each. Saved artifacts persist geometry classes as full module paths, and `rebuild_emulator` imports exactly the stored string, so the old paths must import forever. |
| `background.py` | The BAOSN imposed physics: the legacy cumulative Simpson (verbatim), c/H on the doubled grid, the flat distance conversions, `distance_interpolators`. |
| `syren_base.py` | The syren (symbolic_pofk) base formulas the MPS emulators correct (`base_pklin`, `base_boost`) + `syren_params_from` (the ONE rule mapping resolved parameters to the base's arguments — generator and adapter cannot disagree). |
| `analytics.py` | Closed-form analytic xi (Eisenstein-Hu) to divide out broadband cosmology dependence — the optional rescaling `R`. |

**Model**

| File | Role |
|---|---|
| `activations.py` | Learnable activations: the paper's `H` plus Power / Gated / GatedPower variants; `make_activation` maps a name → factory. |
| `designs/blocks.py` | The small `nn.Module`s models are built from: `Affine`, `ResBlock`, `BinLinear`, `TRFBlock`, `FiLMGenerator`, plus `rescale_kernel_size`. |
| `designs/plain.py` | The full networks: `ResMLP` (baseline; the only design the scalar / cmb / grid / grid2d families accept — the conv/TRF heads assume the 3x2pt eigenbasis geometry), `ResCNN`, `ResTRF`. The factored-IA and NPCE variants are `designs/ia.py` / `designs/pce.py`. |

**Loss & training**

| File | Role |
|---|---|
| `losses/core.py` | chi2 losses on the whitened residual: `CosmolikeChi2` (plain; the `sqrt` / pseudo-Huber / `berhu` / `berhu_capped` mode ladder), `RescaledChi2` / `ResidualBaseChi2` (analytic-R), `ElementWeightedChi2`; `anneal_value`; `make_chi2`. The `_reduce` here is THE shared reduction every family's loss routes through. |
| `losses/scalar.py` | `ScalarChi2` + `make_scalar_chi2`: the standardized-residual chi2 (also wraps the grid and grid2d geometries — their laws live in the geometry, so the loss needs nothing new). |
| `losses/cmb.py` | `AMPLITUDE_LAWS` (`none` / `as_exp2tau`), `CmbDiagonalChi2` / `CmbFactoredChi2` (the imposed-amplitude target), `ResidualRoughness` (the optional band-explicit penalty on short-period residual wiggles), `make_cmb_chi2`. |
| `losses/transfer.py` | `TransferChi2`: a frozen base network under a parallel correction (gain / sum, physical / whitened space). |
| `batching.py` | Memory sizing + the regime-aware loaders (GPU-resident / RAM-stream / memmap-stream) that feed the training loop. |
| `training.py` | Device pick, the `make_model/optimizer/scheduler` factories, `build_run_specs`, the `[default, min, max, kind]` search resolvers, the config validators (`validate_phase_block` / `validate_loss` — now with the `roughness:` sub-block — / `validate_berhu` / `validate_ema`), the per-epoch loop, and `run_emulator`. |

**Orchestration & output**

| File | Role |
|---|---|
| `experiment.py` | `EmulatorExperiment`: config → device → data → geometry → chi2 → spec → train as one reusable object (`from_yaml` / `from_config`). Holds every family's validator (`validate_scalar` / `validate_cmb` / `validate_grid` / `validate_grid2d` / `validate_param_cuts` / `validate_sizes`), the family branches of `from_config` / `build_geometry` (including the fine-tune geometry pins), and the grid2d staging law transform. The drivers compose it. |
| `warmstart.py` | Fine-tune / transfer sources: `load_source` (validate a saved artifact), `extend_input_geometry` (block-extend for new parameters), `pin_output_geometry` (the cosmolike pin; the scalar/cmb/grid/grid2d pins live in their `build_geometry` branches), `build_warm_start` (transfer the weights + prove epoch-0 parity), `anchor_masks`. |
| `scheduling.py` | GPU job balancing (`lpt_assign`, `even_assign`), the spawned worker pool (`run_gpu_pool`), and the `--gpu-pack` VRAM-token machinery. |
| `results.py` | `save_learning_curves` / `save_sweep_table`; `save_emulator` (`.emul` weights + `.h5` record — geometries persisted by `state()` + full cls path); `rebuild_emulator` (the h5-only guarantee; its `info` dict carries the family flags `scalar` / `cmb` / `grid` / `grid2d` + each family's artifact facts, class-guarded). |
| `inference.py` | `EmulatorPredictor`: rebuild a saved emulator and predict — one `predict(params)` for every artifact kind: the dv section, the scalar `{name: value}` dict, the physical C_ell row, the background `{"z", quantity}` function, or the (z, k) law-space surface. Reuses the exact training decode per family. |
| `plotting.py` | Training history, learning-curve overlays, coverage panels, xi curves, and the multipage diagnostics PDF with the per-family pages (`cmb=` / `scalar=` / `grid=`). |
| `diagnostics.py` | Post-training analyses: the family-generic chi2 trio (coverage, local-linear floor, hard directions) + the per-family physical analyses (`cmb_residual_diagnostic`, `scalar_output_diagnostic`, `grid_residual_diagnostic`). |
| `family_drivers.py` | `run_ntrain_sweep` / `run_tune`: the SERIAL per-family loops the thin `sweep_ntrain_<family>_` / `tune_<family>_` drivers call (the multi-GPU pool stays the cosmic-shear drivers' tool). |
| `cocoa.py` | The cocoa project layout: `--root` / `--fileroot` / `--yaml` resolution, output paths. |

**Drivers** (beside `emulator/`; each reads `--root` / `--fileroot` / `--yaml`)

| File | Role |
|---|---|
| `train_single_emulator_cosmic_shear.py` | One training run — cosmic shear, cmb, grid, or grid2d (the data block picks the family); `--diagnostic` writes the multipage PDF with that family's pages. |
| `train_scalar_emulator.py` | One scalar training run; `--diagnostic` adds the scalar pages. |
| `tune_single_emulator_cosmic_shear.py` | Optuna study; multi-GPU via a shared journal-file study. |
| `tune_{scalar,cmb,baosn,mps}_emulator.py` | The per-family Optuna studies (serial, in-memory; `family_drivers.run_tune`). |
| `sweep_ntrain_emulator_cosmic_shear.py` | `f(dchi2 > thr)` vs `N_train`; multi-GPU, LPT-balanced; `--gpu-pack`. |
| `sweep_ntrain_{scalar,cmb,baosn,mps}_emulator.py` | The per-family learning curves (serial; `family_drivers.run_ntrain_sweep`). |
| `sweep_hyperparam_emulator_cosmic_shear.py` | Sweep ONE hyperparameter chosen in the YAML `sweep:` block; multi-GPU. |
| `bakeoff_activation_emulator_cosmic_shear.py` | One learning curve per activation; multi-GPU. |

The naming rule for every new driver is `<verb>_<family>_emulator.py`;
renaming the pre-rule cosmic-shear drivers into it is a recorded POL-1
follow-up (the board configs move with the rename).

**cobaya_theory/** (one thin adapter per artifact kind; each rejects the
other kinds' artifacts loudly, naming the right adapter)

| File | Serves |
|---|---|
| `emul_cosmic_shear.py` | data-vector artifacts → the likelihood's dv (`state["cosmic_shear"]`). |
| `emul_scalars.py` | scalar artifacts → named derived parameters (provides read FROM the artifacts). |
| `emul_cmb.py` | CMB artifacts → the cobaya Cl dict (spectra / lmax / units are artifact facts; `must_provide` refuses beyond-training requests). |
| `emul_baosn.py` | the H(z) + recombination-D_M pair → Hubble + distances, served PIECEWISE by redshift window (the desert between the windows is a loud error, never a bridge); flat-only V1. |
| `emul_mps.py` | the pklin + boost pair → `get_Pk_grid` / `get_Pk_interpolator` (linear + nonlinear) for cosmolike's hybrid mode (`use_emulator: 2`); multiplies the syren base back per the artifacts' stored laws. |

**compute_data_vectors/** (the training-set generators)

| File | Role |
|---|---|
| `generator_core.py` | The shared machinery: the CLI (identical flags for every driver), emcee/uniform sampling, the chain + `.paramnames` + `.ranges` + `.covmat` writers, checkpoint save/load/append, the RAM-aware dv store, the MPI master/worker farm. Drivers subclass `GeneratorCore` and override only the probe whitelist, their train_args keys, the dv store hooks, and `_compute_dvs_from_sample`. |
| `dataset_generator_lensing.py` | cosmolike data vectors (cs / ggl / gc); the core's default single-2D store. |
| `dataset_generator_cmb.py` | CMB spectra: four per-spectrum 2D files (tt / te / ee / pp) from one CAMB pass, phi-phi filled. |
| `dataset_generator_background.py` | H(z) on the SN grid + D_M on the recombination window, one background-only CAMB evaluation per sample, grid sidecars beside the dumps. |
| `dataset_generator_mps.py` | linear P + boost on the (z, k) grids (+ the syren base files when `write_syren_base`), through the Pk_interpolator requirement (the wants-Cl quirk kept verbatim). |
| `compute_cmb_covariance.py` | The Motloch & Hu CMB covariance (eqs 1-7): the Gaussian part always, the lens-induced non-Gaussian terms behind a default-off flag with a 5-point-stencil convergence study; writes the `.npz` the CMB training consumes. |

---

## 4. Change X → edit Y

| To change… | Edit |
|---|---|
| a model architecture | `designs/plain.py` (+ `designs/blocks.py`); the IA / NPCE variants in `designs/ia.py` / `designs/pce.py`; register in `experiment.py`'s `models` |
| an activation function | `activations.py` (register it in `make_activation`) |
| the loss / add a chi2 variant | `losses/core.py` (variants in their family files); the shared reduction is `_reduce` there |
| a new imposed target law | the family geometry's registry + its executor (`losses/cmb.py` for amplitude laws; the geometry itself for grid laws; `syren_base.py` for a cosmology-dependent base) |
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
| a CLI driver (add/modify) | the `<verb>_<family>_emulator.py` beside `emulator/` (compose `EmulatorExperiment`; the serial family loops live in `family_drivers.py`) |
| which hyperparameters are searched | the driver YAML (`[default, min, max, kind]`) + resolvers in `training.py` |
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
| Transfer | `warmstart.py` + `losses/transfer.py` | a frozen trained base under a parallel correction net (cosmolike + CMB data-vector families ONLY — the scope ruling; scalar / grid / grid2d have a permanent forbid). |
| Fine-tuning | `warmstart.py` (`train_args.finetune`) | warm-start from a saved source of the SAME family and geometry; epoch 0 reproduces the source exactly. Supported by EVERY family. |

Removed: the per-bin CNN (its own folder, deleted) — tested; the grouped conv
was absorbed into `rescnn`'s `groups` / `separable` knobs, the per-bin split
lost to a single ResMLP; see git history.

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

- `CmbDiagonalGeometry` — per-multipole whitening by the cosmic-variance error bar; persists spectrum / ell / units / the amplitude-law facts (`from_fiducial` for synthetic fixtures — the ruled `sigma_l = C_fid sqrt(2/(2l+1))`; the training path feeds sigma from the covariance `.npz` through `__init__`).

### `emulator/geometries/grid.py` <a name="apx-geometries_grid"></a>

- `TARGET_LAWS` — `{none, log_offset}`; the law lives INSIDE encode/decode here.
- `GridGeometry` — a function on a stored z grid (`from_targets` applies the law first; persists quantity / units / law / offset / z).

### `emulator/geometries/grid2d.py` <a name="apx-geometries_grid2d"></a>

- `TARGET_LAWS_2D` — `{none, syren_linear, syren_halofit}` (names only: the cosmology-dependent base is the consumer's multiply, through `syren_base.py`).
- `Grid2DGeometry` — a flattened (z-outer) surface standardized in LAW space; persists quantity / units / law / z / k (the stored k IS the thinned grid).

### `emulator/background.py` <a name="apx-background"></a>

The BAOSN imposed physics (one definition for the adapter AND direct scripts).

- `cumulative_simpson(z, y)` — the legacy rule verbatim (even doubled-grid points exact; the odd half-chunk step is a recorded legacy approximation).
- `comoving_distance_grid(z_grid, h_grid)` — c/H cubic onto the doubled grid, Simpson.
- `distance_interpolators(z_grid, h_grid)` — the H / chi / D_A / D_L cubics + the window edge.

### `emulator/syren_base.py` <a name="apx-syren_base"></a>

The syren analytic P(k) base (one definition for the generator, `emul_mps`,
and the gates).

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
- `designs/plain.py` — `ResMLP` (input projection → residual blocks → output projection → Affine), `ResCNN` (gated bins-as-channels 1D-CNN correction in theta order), `ResTRF` (bin-token transformer correction). The head knobs live under YAML `model.cnn` / `model.trf`; see each class docstring for the full knob table and the shape-flow diagrams.
- `designs/ia.py` — `TemplateMLP`, `TemplateResCNN`, `TemplateResTRF` (cosmology-only templates; the amplitudes never enter the network).
- `designs/pce.py` — `PCEEmulator` (the closed-form sparse-Legendre base; loss-owned, not an SGD architecture) + `pce_multi_index`, `pce_design`, `select_lars_loo`.

### `emulator/losses/` (the chi2 family) <a name="apx-losses"></a>

Each loss holds a geometry (composition) and routes through `core.py`'s
shared `_reduce`.

- `losses/core.py` — `anneal_value`; `CosmolikeChi2` (the plain chi2 + the `chi2`/`sqrt`/`sqrt_dchi2`/`berhu`/`berhu_capped` transform ladder); `RescaledChi2` / `ResidualBaseChi2` (analytic-R); `ElementWeightedChi2`; `make_chi2`.
- `losses/ia.py` — `nla_coeffs`, `tatt_coeffs`; `NLAAmpFactoredChi2`, `TemplateFactoredChi2`.
- `losses/pce.py` — `PCEResidualChi2`, `PCERatioChi2`.
- `losses/scalar.py` — `ScalarChi2` + `make_scalar_chi2` (the standardized-residual chi2; also wraps `GridGeometry` and `Grid2DGeometry` — their laws live in the geometry).
- `losses/cmb.py` — `AMPLITUDE_LAWS`; `CmbDiagonalChi2` (plain per-multipole chi2) / `CmbFactoredChi2` (the imposed `C_ell e^{2tau}/A_s` target, reading named columns); `ResidualRoughness` + `configure_roughness` (the optional short-period residual penalty, byte-identical when absent); `make_cmb_chi2`.
- `losses/transfer.py` — `TransferChi2` (a frozen base + a parallel correction; gain/sum, physical/whitened).

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
- `stage_train` / `stage_val` / `pool_size` — stage the sources (the grid2d branch materializes the law-space rows here, `_grid2d_law_rows`); the physical-cut pool size.
- `build_geometry` / `build_specs` — the input/output geometry + chi2 per family (the fine-tune pins live here); the `run_emulator` spec dicts.
- `train` / `run` / `frac_above` — train on the staged data; the full pipeline; the sweep metric.
- `print_design()` — the shared startup banner (family line included), so a stale YAML is caught at launch.

### `emulator/warmstart.py` <a name="apx-warmstart"></a>

Fine-tune / transfer sources (the FTW machinery every family's finetune
routes through).

- `validate_finetune_config(...)` — the config-time whitelist (architecture / activation / loss form inherited; no model: block).
- `resolve_source_root` / `load_source` — resolve + validate a saved source artifact (plain input geometry, rescale none, no chaining).
- `recipe_to_model_opts` / `extend_input_geometry` / `pin_output_geometry` — the recipe → live specs, the block-extension for new parameters, the cosmolike output pin (the other families pin in their `build_geometry` branches).
- `build_warm_start(...)` — transfer the weights (zero-padded extra columns) + prove epoch-0 parity; `transfer_state_dict`, `anchor_masks`, `build_transfer_start`, `_zero_final_linear`.

### `emulator/scheduling.py` <a name="apx-scheduling"></a>

- `lpt_assign` / `even_assign` — GPU job splits.
- `run_gpu_pool(...)` — the spawned worker pool (one process per GPU lane).
- `estimate_train_vram_fraction` / `vram_tokens` — the `--gpu-pack` machinery.

### `emulator/results.py` <a name="apx-results"></a>

- `save_learning_curves` / `save_sweep_table` — plain-text tables.
- `save_emulator(...)` — `.emul` (cpu state_dict) + `.h5` (geometry `state()` groups with their FULL cls paths, histories, raw + resolved config, the model recipe, run-identity attrs; a `pce` / `transfer_base` group when present).
- `rebuild_emulator(path_root, device)` — reconstruct `(model, param_geometry, geometry, info)` from the files alone; `info` carries the family flags (`scalar` / `cmb` / `grid` / `grid2d`) and each family's artifact facts (amplitude law, grid quantity/units/law), class-guarded so one family's facts never smear onto another.

### `emulator/inference.py` <a name="apx-inference"></a>

- `EmulatorPredictor` — rebuild + predict for every artifact kind. `__init__` branches on the rebuilt family (scalar → output names; cmb → spectrum/ell/units + the law-dispatched decoder; grid → quantity/z; grid2d → quantity/z/k; else the dv path with the ia/pce/transfer decoder). `predict(params)` returns the family's natural object (see the class docstring's table); `_build_decoder` / `_build_cmb_decoder` reuse the exact training `chi2fn.decode`.

### `emulator/plotting.py` <a name="apx-plotting"></a>

Figures (colorblind-safe palette, no red/green).

- `plot_history` / `plot_learning_curves` / `plot_sweep_curve` / `plot_diagnostics` — the public figures; `plot_diagnostics` appends the per-family pages when given `cmb=` / `scalar=` / `grid=` (absent = the cosmic-shear PDF, byte-identical).
- `plot_xi` / `dv_to_xi` / `source_param_samples` — xi curves and the coverage triangle.
- `_history_panels` / `_coverage_panels` / `_floor_panel` / `_hard_direction_panels` / `_lcdm_triangle_fig` / `_lnparam_pca_fig` / `_cmb_pages` / `_scalar_pages` / `_grid_pages` / `_finish` / `_save_pages` — the panel and page builders.

### `emulator/diagnostics.py` <a name="apx-diagnostics"></a>

Post-training analyses (each returns a dict the plotting reads).

- `coverage_diagnostic(...)` / `local_linear_floor(...)` / `hard_direction_regression(...)` — the family-generic chi2 trio (they consume only params + per-sample chi2).
- `cmb_residual_diagnostic(...)` — per-multipole residual bands (fractional AND in error-bar units), the worst-cosmology overlay, the high-pass wiggle content the roughness term targets.
- `scalar_output_diagnostic(...)` — per-output truth/prediction/residual tables (physical + standardized) + the bias-hunt inputs.
- `grid_residual_diagnostic(...)` — per-redshift residual bands + (for a Hubble artifact) the derived D_A / D_L bands through the REAL `background.py` pipeline.

### `emulator/family_drivers.py` <a name="apx-family_drivers"></a>

- `add_sweep_args` / `add_tune_args` — the shared per-family CLI flags.
- `run_ntrain_sweep(args, family, out_default)` — the serial N_train learning curve (stage → train per grid point → `save_learning_curves` + the PDF).
- `run_tune(args, family)` — the serial in-memory Optuna study (TPE seeded, trial 0 warm-started from the YAML defaults).

### drivers (beside `emulator/`) <a name="apx-drivers"></a>

Each `main()` reads `--root` / `--fileroot` / `--yaml`.

- `train_single_emulator_cosmic_shear.py` — one training run (any dv-shaped family) + the diagnostics PDF.
- `train_scalar_emulator.py` — one scalar run + the diagnostics PDF.
- `tune_single_emulator_cosmic_shear.py` — the multi-GPU journal study; `tune_{scalar,cmb,baosn,mps}_emulator.py` — the serial per-family studies.
- `sweep_ntrain_emulator_cosmic_shear.py` — the multi-GPU learning curve; `sweep_ntrain_{scalar,cmb,baosn,mps}_emulator.py` — the serial per-family curves.
- `sweep_hyperparam_emulator_cosmic_shear.py` — one YAML-chosen knob.
- `bakeoff_activation_emulator_cosmic_shear.py` — one curve per activation.
