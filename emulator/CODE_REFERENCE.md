# Python code reference for the `emulator/` package

This page is for readers who need to change the Python package. It maps a
requested change to its first file, explains the model variants, and lists the
main functions and classes.

For an explanation of emulator models and training choices, start with the
[short package guide](README.md). For YAML settings, use the
[example-YAML guide](../example_yamls/README.md).

## Contents

1. [What does each group of files do?](#3-what-each-file-does)
2. [Which file should a code change edit?](#4-change-x--edit-y)
3. [Which model variants have separate files?](#5-variants)
4. [Which functions and classes does each file provide?](#6-every-files-functions)

---

## Questions about the Python code

### D1. What does each group of files do? <a id="3-what-each-file-does"></a>

The package separates scientific conversions from the neural model and the
training loop. This table is the shortest code map:

| Question | Files to inspect |
|---|---|
| Which rows and columns enter a run? | `parameter_table.py`, `data_staging.py` |
| How are parameters and outputs converted? | `geometries/` |
| Which analytic physics is applied outside the network? | `background.py`, `syren_base.py`, `analytics.py` |
| How is the neural model assembled? | `activations.py`, `designs/` |
| How is scientific error calculated? | `losses/` |
| How are batches loaded and weights updated? | `batching.py`, `training.py` |
| How does a YAML file become one experiment? | `experiment.py` |
| Where are project, YAML, and output paths resolved? | `cocoa.py` |
| How is a saved model reused? | `warmstart.py`, `results.py`, `inference.py` |
| How are figures and numerical checks made? | `plotting.py`, `diagnostics.py` |
| How are multi-run jobs divided among devices? | `scheduling.py`, `family_drivers.py`, `studies/` |

Only cosmic-shear training imports CosmoLike, through
`geometries/output.py`. The other training families still require PyTorch,
NumPy, SciPy, YAML, HDF5, and psutil. Some plots require Matplotlib and
GetDist.

The commands that call this package sit one folder above `emulator/`. Other
folders have their own guides:

| Folder | Guide |
|---|---|
| `example_yamls/` | [copying and editing settings](../example_yamls/README.md) |
| `compute_data_vectors/` | [creating scientific tables](../compute_data_vectors/README.md) |
| `cobaya_theory/` | [using saved models in Cobaya](../cobaya_theory/README.md) |
| `syren/` | [the analytic matter-power formulas](../syren/README.md) |
| `ai/gates/` | [optional AI-development checks](../ai/gates/README.md) |

### D2. Which file should a code change edit? <a id="4-change-x--edit-y"></a>

| To change… | Edit |
|---|---|
| a neural-network design | `designs/plain.py` (+ `designs/blocks.py`); the IA / NPCE variants in `designs/ia.py` / `designs/pce.py`; register in `experiment.py`'s `models` |
| an activation function | `activations.py` (register it in `make_activation`) |
| the loss / add a chi2 variant | `losses/core.py` (variants in their family files); the shared reduction is `_reduce` there |
| a new imposed target law | the family geometry's registry + its executor (`losses/cmb.py` for amplitude laws; the geometry itself for grid laws; `syren_base.py` for a cosmology-dependent base) |
| update the Syren base formulas | update `syren/` as described in [its guide](../syren/README.md), then retrain the MPS artifacts; existing base tables still contain the formula used to create them |
| how parameters are whitened (input) | `geometries/parameter.py` |
| dv whitening / cosmolike reading (output) | `geometries/output.py` |
| a new output family | a geometry module with `state` and `from_state`, a loss or `ScalarChi2`, the family branches in `experiment.py` and `inference.py`, a Cobaya adapter, and focused tests; scalar, CMB, grid, and grid2d are examples |
| data loading, physical cuts, and preparing selected rows for training | `data_staging.py` |
| restrict the training pool by a density window | the `data.param_cuts:` sub-block; a new window is one row in `data_staging.phys_cut_idx`'s table |
| GPU memory use and batch loading | `batching.py` |
| the optimizer/scheduler build or the training loop | `training.py` |
| how settings assemble a complete run | `experiment.py` |
| fine-tuning / transfer mechanics | `warmstart.py` (+ the family pin in `experiment.build_geometry`) |
| what an artifact stores | update both save and rebuild in `results.py`, plus the geometry's `state()` when its saved facts change |
| how a saved emulator is served | the predictor branch in `inference.py`, then the thin adapter in `cobaya_theory/` |
| a CLI driver (add/modify) | the `<family>_<verb>_emulator.py` beside `emulator/` (tune/sweep family versions and the CMB/BAOSN/MPS trainers are thin wrappers over the cosmic-shear mains; scalar train is the standalone, data-vector-free exception) |
| which hyperparameters are searched | the driver YAML (`[default, min, max, kind]`) + resolvers in `training.py` |
| the saved description, journal-file check, or stable name of a tuning study | `studies/manifest.py`, `studies/manifest_digest.py`, `studies/implementation.py`, or `studies/name.py` |
| multi-GPU balancing | `scheduling.py` |
| the training-set sampling / checkpoints / MPI farm | `compute_data_vectors/generator_core.py` (all four generators inherit) |
| one generator's physics | that generator's `_compute_dvs_from_sample` only |
| the output file format | `results.py` |
| a plot | `plotting.py` |
| a diagnostic | `diagnostics.py` |
| analytic preprocessing | `analytics.py` |

---

### D3. Which model variants have separate files? <a id="5-variants"></a>

The design/loss family folders each carry the main path plus its variants,
one file per variant. Each pairs a design (or a frozen base) with its own
loss.

| Variant | Design + loss | What it is |
|---|---|---|
| NPCE | `designs/pce.py` + `losses/pce.py` | a sparse-Legendre polynomial-chaos base plus a neural refiner. |
| Factored IA | `designs/ia.py` + `losses/ia.py` | emulate cosmology-only templates and apply the IA-amplitude polynomial in closed form (the amplitudes never enter the network). |
| Transfer | `warmstart.py` + `losses/transfer.py` | a trained base under a parallel correction network. The base stays frozen in ordinary transfer. CosmoLike alone may add a later `refine` stage that adjusts both; CMB, grid, and grid2d reject that stage. Scalar artifacts use fine-tuning instead. |
| Fine-tuning | `warmstart.py` (`train_args.finetune`) | warm-start from a compatible saved source. The rebuilt function must reproduce the source within the configured numerical agreement limit, and newly added inputs must initially have exactly zero influence. Supported by every family. |

A singular-value decomposition (SVD) rewrites a matrix as ordered numerical
directions. The grid2d NPCE fit materializes the thinned target on the
accelerator and runs a dense float64 SVD on the CPU. Use this fit only when
the thinned target and the SVD matrix fit their separate memory budgets.

---

### D4. Which functions and classes does each file provide? <a id="6-every-files-functions"></a>

This index names the main Python objects. A **function** performs one named
operation. A **class** defines an object that stores data and related
operations. A **method** is a function attached to a class.

For arguments, array shapes, return values, and errors, read the docstring
inside the named file.

#### `emulator/parameter_table.py` <a name="apx-parameter_table"></a>

Reads a GetDist `.paramnames` file, which declares each table column's name
and whether it is an input or derived output.

- `ResolvedParameterTable` — fixed record containing two-dimensional
  `float32` input and output arrays, the parsed declarations, and the selected
  `.paramnames` path.
- `resolve_parameter_table(params_path, input_names, output_names=())` — require the exact-stem or numeric-chain-root `.paramnames`, validate every declared name, role, and numeric width, then select inputs and derived outputs by name and requested order.

#### `emulator/data_staging.py` <a name="apx-data_staging"></a>

Builds the dictionaries that connect parameter rows with matching target
rows. A NumPy **memory map** reads selected parts of an array from disk
without first loading the whole file into RAM.

- `load_source(...)` — validate the named parameter table before opening the data-vector memory map, apply parameter cuts, keep `N_train` rows, and return `{C, dv, idx, dump_rows (+ means)}`. `dump_rows` gives the original disk row for each selected row used by training.
- `load_scalar_source(...)` — the scalar sibling: inputs AND outputs are named columns of one parameter `.txt` (the getdist `.paramnames` sidecar locates them).
- `stage_source(C, dv, idx, ram_frac)` — copy selected rows into RAM when they
  fit; otherwise keep the target as a memory map.
- `phys_cut_idx(...)` — keep the rows inside every active physical-density window; returns `(kept_idx, report)`.
- `stream_chunks` / `stream_stats` / `param_stats` — streamed per-column stats (the dump never loads whole).
- `read_param_names(covmat_path, comment)` — parameter names from the covmat header line.

#### `emulator/geometries/parameter.py` <a name="apx-geometries_parameter"></a>

Input side: physical parameters to the model's numerical coordinates.
**Whitening** centers the values, rotates them into covariance eigenvectors,
and scales those directions to comparable size.

- `ParamGeometry` — center, rotate into the parameter-covariance eigenbasis, unit-scale (`from_covmat` / `from_state` / `state`; `whiten`/`unwhiten`, `encode`/`decode`).
- `LogParamGeometry` — whitens in log space for multiplicative params.
- `AmplitudeFactorGeometry` — whiten every parameter except the IA amplitude(s), append them raw for the loss's closed-form combine.

#### `emulator/geometries/output.py` <a name="apx-geometries_output"></a>

The 3x2pt output side converts between the physical data vector and the
whitened entries kept by the likelihood. It also holds the covariance used
to calculate chi-square. This is the only file that imports CosmoLike.

- `DataVectorGeometry` — the base geometry for one probe (`from_cosmolike` / `from_state` / `state`; `squeeze`/`unsqueeze`; `whiten`/`unwhiten`, `encode`/`decode`).
- `DiagonalGeometry` — whiten by the marginal sigma only (theta order kept, for a CNN).
- `BlockDiagonalGeometry` — whiten each tomographic bin by its own sub-block.
- `build_shear_angle_map(geom, ...)` — attach per-element theta / source-z / xi± branch / per-bin sizes.

#### `emulator/geometries/scalar.py` <a name="apx-geometries_scalar"></a>

- `ScalarGeometry` — per-output standardization over named derived parameters (`from_targets` with the un-standardizable guard; `from_state` / `state`; `encode`/`decode`; `dest_idx`/`total_size` the trivial identity).

#### `emulator/geometries/cmb.py` <a name="apx-geometries_cmb"></a>

- `CmbDiagonalGeometry` — subtracts the target center and divides by the persisted `sigma_<spectrum>` from the covariance `.npz`; persists the spectrum, multipole grid, units, and amplitude-law facts. `from_fiducial` is a synthetic-construction helper; the training path passes the stored scale through `__init__`.
- `attach_head_coords()` — describe one spectrum bin with multipole as its
  coordinate for CNN and transformer heads. Calling it twice gives the same
  result; training and model reload both call it.

#### `emulator/geometries/grid.py` <a name="apx-geometries_grid"></a>

- `TARGET_LAWS` — the accepted target conversions, `{none, log_offset}`;
  `GridGeometry` applies and reverses the selected conversion.
- `GridGeometry` — a function on a stored z grid (`from_targets` applies the law first; persists quantity / units / law / offset / z).
- `attach_head_coords()` — the heads' split: one bin, coordinate = z.

#### `emulator/geometries/grid2d.py` <a name="apx-geometries_grid2d"></a>

- `TARGET_LAWS_2D` — `{none, syren_linear, syren_halofit}` (names only: the cosmology-dependent base is the consumer's multiply, through `syren_base.py`).
- `Grid2DGeometry` stores a function on a two-dimensional grid. Redshift is
  the outer index and wavenumber is the inner index.
  - The geometry applies the selected target law and standardizes the flattened
    values.
  - Its saved state records the quantity, units, law, both axes, and a
    `const_mask`.
  - A false mask entry leaves that network output active. A true entry returns
    the stored training constant, such as the low-wavenumber $B=1$ tail of a
    boost.
  - An all-false mask records that no entries are pinned. A surface that is
    constant everywhere remains an error because it indicates a dead data
    dump.
- `attach_head_coords()` — describe one bin for each redshift slice, with the
  stored wavenumbers inside that bin.

#### `emulator/background.py` <a name="apx-background"></a>

Calculates cosmological distances from $H(z)$ for the background adapter and
direct Python use.

- `cumulative_simpson(z, y)` — integrate on the doubled grid; even points are exact for cubic input, and each odd point uses the one-interval rule `h/12*(5,8,-1)`, which is exact for quadratic input.
- `comoving_distance_grid(z_grid, h_grid)` — c/H cubic onto the doubled grid, Simpson.
- `distance_interpolators(z_grid, h_grid)` — the H / chi / D_A / D_L cubics + the window edge.

#### `emulator/syren_base.py` <a name="apx-syren_base"></a>

Calculates the analytic Syren $P(k)$ base used by the data generator,
`emul_mps`, and the tests. The formulas come from the repository's stored
copy under `syren/`; [its guide](../syren/README.md) records their source and
local differences.

- `syren_params_from(params)` — map resolved parameters to the Syren arguments. It accepts `As` or `As_1e9`; an absent equation of state means LCDM.
- `base_pklin(...)` / `base_boost(...)` — calculate the analytic linear-power and nonlinear-boost bases with the units stated in `syren/README.md`.

#### `emulator/analytics.py` <a name="apx-analytics"></a>

Analytic xi rescaling `R` (Eisenstein-Hu zero-baryon preprocessor).

- `_analytic_R(...)` / `analytic_shape_ratio(...)` / `rescale_xi(...)`.

#### `emulator/activations.py` <a name="apx-activations"></a>

Learnable activations for the ResBlock `act` slot.

- `activation_fcn` — the paper's `H` (a learnable identity↔Swish interpolation).
- `GatedActivation` / `PowerGatedActivation` / `GatedPowerActivation` — generalizations.
- `make_activation(name, n_gates)` — map a name to a factory `act(dim) -> module`.

#### `emulator/designs/` (the model family) <a name="apx-designs"></a>

Shared `designs/blocks.py`, the plain models in `designs/plain.py`, and the
factored-IA and NPCE variants in `designs/ia.py` and `designs/pce.py`. The
class flags the config layer reads:
`head_block` (which correction head, None = trunk-only), `factored` (IA
amplitudes appended raw), `needs_bins` (wants the shear-angle map).

- `designs/blocks.py` — `Affine`, `FeatureAffine`, `make_norm`, `ResBlock`, `BinLinear`, `TRFBlock`, `FiLMGenerator`, `rescale_kernel_size`.
- `designs/plain.py` — `ResMLP` (input projection → residual blocks → output projection → Affine), `ResCNN` (gated bins-as-channels 1D CNN correction in theta order), `ResTRF` (bin-token transformer correction). The head knobs live under YAML `model.cnn` / `model.trf`; see each class docstring for the full knob table and the shape-flow diagrams.
- `designs/ia.py` — `TemplateMLP`, `TemplateResCNN`, `TemplateResTRF` (cosmology-only templates; the amplitudes never enter the network).
- `designs/pce.py` — `PCEEmulator` (the closed-form sparse-Legendre base; loss-owned, not an SGD architecture) + `pce_multi_index`, `pce_design`, `select_lars_loo`.

#### `emulator/losses/` (the chi2 family) <a name="apx-losses"></a>

Each loss holds a geometry and routes through `losses/core.py`'s
shared `_reduce`.

- `losses/core.py` — `anneal_value`; `CosmolikeChi2` (the plain chi2 + the `chi2`/`sqrt`/`sqrt_dchi2`/`berhu`/`berhu_capped` transform ladder); `RescaledChi2` / `ResidualBaseChi2` (analytic-R); `ElementWeightedChi2`; `make_chi2`.
- `losses/ia.py` — `nla_coeffs`, `tatt_coeffs`; `NLAAmpFactoredChi2`, `TemplateFactoredChi2`.
- `losses/pce.py` — `PCEResidualChi2`, `PCERatioChi2` (the cosmolike forms); `PCEResidualDiagChi2` (the family-wide residual form over the elementwise-whitened geometries — cmb law-none / grid / grid2d / scalar; roughness composes).
- `losses/scalar.py` — `ScalarChi2` + `make_scalar_chi2` (the standardized-residual chi2; also wraps `GridGeometry` and `Grid2DGeometry` — their laws live in the geometry).
- `losses/cmb.py` — `AMPLITUDE_LAWS`; `CmbDiagonalChi2` (plain per-multipole chi-square); `CmbFactoredChi2` (the imposed `C_ell e^{2tau}/A_s` target); `ResidualRoughness` and `configure_roughness` (the optional short-period residual penalty); `make_cmb_chi2`.
- `losses/transfer.py` — `TransferChi2` combines a base with a parallel correction in physical or converted coordinates; the base is frozen in the ordinary pass and may be live during CosmoLike `refine`. `TransferDiagChi2` accepts plain diagonal-family bases and reports an error for unsupported refine or roughness combinations.

#### `emulator/batching.py` <a name="apx-batching"></a>

Memory sizing and data loaders for three storage choices: all data on the GPU,
arrays in computer RAM loaded in groups, or disk-backed memory maps loaded in
groups.

- `compute_batch_size_bytes` / `compute_model_size_bytes` / `batches_per_load` — memory estimates.
- `_build_loaders_one(...)` / `build_loaders(...)` — pick a regime per source; return the loaders the loop consumes.

#### `emulator/training.py` <a name="apx-training"></a>

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

#### `emulator/experiment.py` <a name="apx-experiment"></a>

`EmulatorExperiment`: the whole setup as one reusable object.

- `from_yaml` / `from_config` — build from a YAML file or parsed dictionary; `from_config` selects the scalar, CMB, grid, grid2d, or cosmic-shear branch.
- `validate_param_cuts` / `validate_sizes` / `validate_scalar` / `validate_cmb` / `validate_grid` / `validate_grid2d` / `validate_transfer` — the pure data-block validators, one per concern.
- `resolve_phase_args` / `validate_sweep_paths` — the two-phase schedule resolution.
- `_head_activation_spec` / `_resolve_head_activation` / `_activation_flag_notice` / `_pinned_head_warning` — the activation setting for each head.
- `stage_train` / `stage_val` / `pool_size` — prepare the sources and report their legal row ceiling (the full named table with no cuts, or the physical-window survivor count). The grid2d branch thins the k axis, forms law-space rows in bounded chunks, computes moments from the stored float32 payload, and uses an experiment-owned temporary memmap when a full copy in RAM would exceed its memory budget.
- `build_geometry` / `build_specs` — the input/output geometry + chi2 per family (the fine-tune pins live here); the setting dictionaries passed to `run_emulator`.
- `train` / `run` / `frac_above` — train on the selected data; the complete training process; the sweep metric.
- `print_design()` — the shared startup banner (family line included), so a stale YAML is caught at launch.

#### `emulator/warmstart.py` <a name="apx-warmstart"></a>

Fine-tune / transfer sources (the warm-start machinery every family's finetune
routes through).

- `validate_finetune_config(...)` — the config-time whitelist (architecture / activation / loss form inherited; no model: block).
- `resolve_source_root` / `load_source` — resolve + validate a saved source artifact (plain input geometry, rescale none, no chaining).
- `recipe_to_model_opts` / `extend_input_geometry` / `pin_output_geometry` — the recipe → live specs, the block-extension for new parameters, the cosmolike output pin (the other families pin in their `build_geometry` branches).
- `build_warm_start(...)` — transfer the weights (zero-padded extra columns) + prove epoch-0 parity; `transfer_state_dict`, `anchor_masks`, `build_transfer_start`, `_zero_final_linear`.

#### `emulator/scheduling.py` <a name="apx-scheduling"></a>

- `lpt_assign` / `even_assign` — GPU job splits.
- `run_gpu_pool(...)` — start and join one worker process per selected GPU.
- `estimate_train_vram_fraction` / `vram_tokens` — the `--gpu-pack` machinery.

#### `emulator/studies/` (saved Optuna-study description) <a name="apx-studies"></a>

- `studies/implementation.py` — `study_implementation_identity`, backed by
  the saved version list for scientific tuning components.
- `studies/manifest.py` — `build_study_manifest`, `bind_study_manifest`, and
  `require_worker_identity`; record the scientific inputs and confirm that a
  worker is writing to the intended journal file.
- `studies/manifest_digest.py` — `canonical_json`, `manifest_digest`, and
  `file_digest`; strict canonical bytes and SHA-256 identities.
- `studies/name.py` — `resolve_study_name`; stable names for all five emulator
  families, independent of wrapper filenames.

#### `emulator/results.py` <a name="apx-results"></a>

- `save_learning_curves` / `save_sweep_table` — plain-text tables.
- `save_emulator(...)` — writes a CPU-normalized PyTorch `state_dict` to `.emul` and geometry states, histories, resolved config, model recipe, and optional PCE or transfer-base records to `.h5`.
- `rebuild_emulator(path_root, device)` — reconstructs `(model, param_geometry, geometry, info)` from the pair. `map_location=device` selects the destination, so loading does not require the accelerator used during saving. `info` carries the family flags and family-specific artifact facts.

#### `emulator/inference.py` <a name="apx-inference"></a>

- `EmulatorPredictor` — rebuild and predict for every saved family. Initialization selects output names and coordinates from the artifact, then builds the same decoder used during training. `_build_diag_decoder` combines a saved PCE base with its neural residual when the artifact contains a `pce` group. `predict(params)` returns the object listed in the class docstring.

#### `emulator/plotting.py` <a name="apx-plotting"></a>

Figures (colorblind-safe palette, no red/green).

- `plot_history` / `plot_learning_curves` / `plot_sweep_curve` / `plot_diagnostics` — make the public figures. `plot_diagnostics` adds family-specific pages when given `cmb=`, `scalar=`, `grid=`, or `grid2d=`; with none of these it makes the cosmic-shear PDF.
- `plot_xi` / `dv_to_xi` / `source_param_samples` — xi curves and the coverage triangle.
- `_history_panels` / `_coverage_panels` / `_floor_panel` / `_hard_direction_panels` / `_lcdm_triangle_fig` / `_lnparam_pca_fig` / `_cmb_pages` / `_scalar_pages` / `_grid_pages` / `_finish` / `_save_pages` — the panel and page builders.

#### `emulator/diagnostics.py` <a name="apx-diagnostics"></a>

Post-training analyses (each returns a dict the plotting reads).

- `coverage_diagnostic(...)` / `local_linear_floor(...)` / `hard_direction_regression(...)` — the family-generic chi2 trio (they consume only params + per-sample chi2).
- `cmb_residual_diagnostic(...)` — per-multipole residual bands (fractional AND in error-bar units), the worst-cosmology overlay, the high-pass wiggle content the roughness term targets.
- `scalar_output_diagnostic(...)` — per-output truth/prediction/residual tables (physical + standardized) + the bias-hunt inputs.
- `grid_residual_diagnostic(...)` — per-redshift residual bands + (for a Hubble artifact) the derived D_A / D_L bands through the REAL `background.py` pipeline.
- `grid2d_residual_diagnostic(...)` — the matter-power (z, k) residual surfaces in LAW space (under a syren law the residual = ln(P_pred / P_truth), the base cancels): median-|residual| + worst cosmology + per-k bands at three redshifts.

#### `emulator/family_drivers.py` <a name="apx-family_drivers"></a>

- `set_by_path(train_args, path, value)` / `read_sweep_block(cfg)` + `SWEEPABLE_TOP_KEYS` / `ACTIVATION_PATHS` — the one definition of the `sweep:` block, imported by the cosmic-shear one-knob driver (the family drivers are wrappers over `main(prog, family)`, so they inherit it).

#### drivers (beside `emulator/`) <a name="apx-drivers"></a>

Each `main()` reads `--root` / `--fileroot` / `--yaml`.

- `cosmic_shear_train_emulator.py` — one training run (any dv-shaped family) + the diagnostics PDF.
- `scalar_train_emulator.py` — one scalar run + the diagnostics PDF.
- `cmb_train_emulator.py` / `baosn_train_emulator.py` / `mps_train_emulator.py` — the thin family wrappers (`main(prog, family)` + `require_family_block`).
- `cosmic_shear_tune_emulator.py` — the Optuna driver (serial or the multi-GPU journal study); `{scalar,cmb,baosn,mps}_tune_emulator.py` — thin family wrappers over its `main(prog, family)`.
- `cosmic_shear_sweep_ntrain_emulator.py` — the learning-curve driver (multi-GPU, LPT, `--gpu-pack`); `{scalar,cmb,baosn,mps}_sweep_ntrain_emulator.py` — thin family wrappers.
- `cosmic_shear_sweep_hyperparam_emulator.py` — one YAML-chosen knob (multi-GPU); `{scalar,cmb,baosn,mps}_sweep_hyperparam_emulator.py` — thin family wrappers.
- `cosmic_shear_bakeoff_activation_emulator.py` — one curve per activation.
