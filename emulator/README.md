# The emulator/ package: code map

How to run the drivers and how to configure the YAML live in
[`../README.md`](../README.md); this file maps the package internals — how
it is laid out, what each file does, and where to edit for a given change.

## Contents

1. [Layout](#1-layout)
2. [What each file does](#2-what-each-file-does)
3. [Change X → edit Y](#3-change-x--edit-y)
4. [Variants](#4-variants)
5. [Every file's functions](#5-every-files-functions)
    1. [`data_staging.py`](#apx-data_staging)
    2. [`geometries_parameter.py`](#apx-geometries_parameter)
    3. [`geometries_output.py`](#apx-geometries_output)
    4. [`analytics.py`](#apx-analytics)
    5. [`activations.py`](#apx-activations)
    6. [`emulator_designs_building_blocks.py`](#apx-building_blocks)
    7. [`emulator_designs.py`](#apx-emulator_designs)
    8. [`loss_functions.py`](#apx-loss_functions)
    9. [`batching.py`](#apx-batching)
    10. [`training.py`](#apx-training)
    11. [`experiment.py`](#apx-experiment)
    12. [`scheduling.py`](#apx-scheduling)
    13. [`results.py`](#apx-results)
    14. [`plotting.py`](#apx-plotting)
    15. [`diagnostics.py`](#apx-diagnostics)
    16. [`PCE/`](#apx-pce)
    17. [`IA/`](#apx-ia)
    18. [drivers](#apx-drivers)

---

## 1. Layout

```
emulator/                              the library (pure torch, except geometries_output)
  data_staging.py                      load dumps -> "source" dicts; the physical cut
  geometries_parameter.py              INPUT whitening (params -> network input)
  geometries_output.py                 OUTPUT geometry + chi2 covariance (imports cosmolike)
  analytics.py                         analytic xi rescaling R (optional preprocessing)
  activations.py                       learnable activations (H + variants)
  emulator_designs_building_blocks.py  Affine, ResBlock, BinLinear, TRFBlock, FiLMGenerator
  emulator_designs.py                  ResMLP, ResCNN, ResTRF
  loss_functions.py                    chi2 losses + make_chi2
  batching.py                          memory sizing + regime-aware data loaders
  training.py                          build model/opt/sched, training loop, run_emulator
  experiment.py                        EmulatorExperiment: the whole setup as one object
  scheduling.py                        GPU job balancing + worker pool + VRAM packing
  results.py                           save_learning_curves; save_emulator (.emul + .h5)
  plotting.py                          history / learning-curve / coverage / xi plots
  diagnostics.py                       coverage, local-linear floor, hard-direction fits
  PCE/  IA/                            experimental variants (section 4)

train_single_*.py                      CLI: one training run (+ optional diagnostics PDF)
tune_single_*.py                       CLI: Optuna hyperparameter search (multi-GPU)
sweep_ntrain_*.py                      CLI: f(dchi2 > thr) vs N_train   (multi-GPU)
sweep_hyperparam_*.py                  CLI: sweep ONE YAML-chosen knob  (multi-GPU)
bakeoff_activation_*.py                CLI: one curve per activation    (multi-GPU)
example_yamls/                         template YAMLs; copy one into a project's --fileroot
```

The driver scripts sit beside `emulator/` (no `driver/` subfolder): launching one
puts its own folder on `sys.path`, so `import emulator` resolves with no path
setup. In a cocoa install this folder is
`external_modules/code/emulators/emultrf/dev/`; run the drivers from `$ROOTDIR`.

Only `geometries_output.py` imports cosmolike, so training runs on a machine
with a working Cocoa installation; the library everywhere else is pure PyTorch
and reviewable anywhere.

---

## 2. What each file does

**Data & geometry**

| File | Role |
|---|---|
| `data_staging.py` | On-disk dumps → in-memory "source" dicts; streaming per-column stats; the physical density windows (`omega_b h^2` bound, plus the optional `omegam^2 h^2` / `omegamh2` / `omegamh2·n_s` windows). Memmaps the dv dump (never loads it whole). |
| `geometries_parameter.py` | Input whitening: `ParamGeometry` (center + rotate into the covmat eigenbasis + unit-scale), `LogParamGeometry`, and the IA-factoring `AmplitudeFactorGeometry`. |
| `geometries_output.py` | Output side: `DataVectorGeometry` (squeeze to unmasked entries, whiten, own the chi2 `Cinv`), `DiagonalGeometry` (theta order, for a CNN), `BlockDiagonalGeometry`, `build_shear_angle_map`. **Only file importing cosmolike.** |
| `analytics.py` | Closed-form analytic xi (Eisenstein-Hu) to divide out broadband cosmology dependence — the optional rescaling `R`. |

**Model**

| File | Role |
|---|---|
| `activations.py` | Learnable activations: the paper's `H` plus Power / Gated / GatedPower variants; `make_activation` maps a name → factory. |
| `emulator_designs_building_blocks.py` | The small `nn.Module`s models are built from: `Affine`, `ResBlock`, `BinLinear`, `TRFBlock`, `FiLMGenerator`, plus `rescale_kernel_size`. |
| `emulator_designs.py` | The full networks: `ResMLP` (baseline), `ResCNN` (ResMLP trunk + a gated 1D-CNN correction in theta order), and `ResTRF` (a bin-token transformer correction head). |

**Loss & training**

| File | Role |
|---|---|
| `loss_functions.py` | chi2 losses on the whitened residual: `CosmolikeChi2` (plain; the `sqrt` / pseudo-Huber / `berhu` / `berhu_capped` mode ladder), `RescaledChi2` / `ResidualBaseChi2` (analytic-R), `ElementWeightedChi2`; `anneal_value` (the shared trim / focus / berhu-blend / EMA-horizon schedule); `make_chi2`. |
| `batching.py` | Memory sizing + the regime-aware loaders (GPU-resident / RAM-stream / memmap-stream) that feed the training loop. |
| `training.py` | Device pick, the `make_model/optimizer/scheduler` factories, `build_run_specs`, the `[default, min, max, kind]` search resolvers, the config validator / derivation layer (`validate_phase_block` / `validate_loss` / `validate_berhu` / `validate_ema`, `derive_eval_bs` / `derive_ema_beta`), the per-epoch loop, and `run_emulator`. |

**Orchestration & output**

| File | Role |
|---|---|
| `experiment.py` | `EmulatorExperiment`: config → device → data → geometry → chi2 → spec → train as one reusable object (`from_yaml` / `from_config`). The drivers compose it. |
| `scheduling.py` | GPU job balancing (`lpt_assign` by cost, `even_assign` round-robin), the spawned worker pool (`run_gpu_pool`: one process per GPU lane, per-GPU job queues), and the `--gpu-pack` VRAM-token machinery (`estimate_train_vram_fraction`, `vram_tokens`). |
| `results.py` | `save_learning_curves` / `save_sweep_table`: `np.loadtxt`-friendly plain-text tables. `save_emulator`: a trained run as `.emul` (weights, cpu state_dict) + `.h5` (whitening geometries, histories, config). |
| `plotting.py` | Training history, learning-curve overlays, coverage panels, xi curves. |
| `diagnostics.py` | Post-training analyses: coverage (kNN distance vs error), the local-linear data floor, the hard-direction regression. |

**Drivers** (beside `emulator/`; each reads `--root` / `--fileroot` / `--yaml`)

| File | Role |
|---|---|
| `train_single_emulator_cosmic_shear.py` | One training run; `--diagnostic` writes a multipage PDF. |
| `tune_single_emulator_cosmic_shear.py` | Optuna study over the YAML's `[default, min, max, kind]` ranges; multi-GPU via a shared journal-file study (`--n-gpus`, `--journal`). |
| `sweep_ntrain_emulator_cosmic_shear.py` | `f(dchi2 > thr)` vs `N_train`; multi-GPU, LPT-balanced; `--gpu-pack` co-locates small points on big cards. |
| `sweep_hyperparam_emulator_cosmic_shear.py` | Sweep ONE hyperparameter chosen in the YAML `sweep:` block (any dotted `train_args` path, e.g. `bs`, `lr.lr_base`, `model.cnn.film`, `model.activation`); multi-GPU, `--gpu-pack`. |
| `bakeoff_activation_emulator_cosmic_shear.py` | One learning curve per activation; multi-GPU, split by activation. |

---

## 3. Change X → edit Y

| To change… | Edit |
|---|---|
| a model architecture | `emulator_designs.py` (+ `emulator_designs_building_blocks.py`) |
| an activation function | `activations.py` (register it in `make_activation`) |
| the loss / add a chi2 variant | `loss_functions.py` |
| how parameters are whitened (input) | `geometries_parameter.py` |
| dv whitening / cosmolike reading (output) | `geometries_output.py` |
| data loading / the physical cut / staging | `data_staging.py` |
| restrict the training pool by a density window | the `data.param_cuts:` sub-block keys (`omegabh2_hi` required, `omegabh2_lo`, `omegam2h2_*`, `omegamh2_*`, `omegamh2ns_*`); a new window is one row in `data_staging.phys_cut_idx`'s table |
| the GPU-memory regime / batching | `batching.py` |
| the optimizer/scheduler build or the training loop | `training.py` |
| the end-to-end setup wiring | `experiment.py` |
| a CLI driver (add/modify) | `*_emulator_cosmic_shear.py` (beside `emulator/`; compose `EmulatorExperiment`) |
| which hyperparameters are searched | the driver YAML (`[default, min, max, kind]`) + resolvers in `training.py` |
| multi-GPU balancing | `scheduling.py` |
| the output file format | `results.py` |
| a plot | `plotting.py` |
| a diagnostic | `diagnostics.py` |
| analytic preprocessing | `analytics.py` |

---

## 4. Variants

Each subfolder mirrors the two-file shape (`emulator_designs.py` +
`loss_functions.py`) and is an experiment, not the main path.

| Folder | What it is |
|---|---|
| `PCE/` | NPCE: a sparse-Legendre polynomial-chaos base plus a neural refiner. |
| `IA/` | Factored intrinsic alignment: emulate cosmology-only templates and apply the IA-amplitude polynomial in closed form (the amplitudes never enter the network). |

Removed: the per-bin CNN (`parallel/`) — tested; the grouped conv was
absorbed into `rescnn`'s `groups` / `separable` knobs, the per-bin split
lost to a single ResMLP; removed — see git history.

---

## 5. Every file's functions

One line per function / class / method. For full detail, read the docstring in
the file itself; this is the index.

### `emulator/data_staging.py` <a name="apx-data_staging"></a>

Turns on-disk dumps into in-memory "source" dicts.

- `load_source(...)` — orchestrator: memmap the dv, load + cut the params, keep `N_train` rows, stage, return `{C, dv, idx (+ means)}`.
- `stage_source(C, dv, idx, ram_frac)` — materialize the used rows in RAM if they fit, else keep the memmap (reindex local).
- `phys_cut_idx(C, idx, names, omegabh2_hi, omegabh2_lo, omegam2h2_lo/hi, omegamh2_lo/hi, omegamh2ns_lo/hi, param_file)` — keep the rows inside every active physical-density window (a small quantity table, one row per window: `omega_b h^2` in `(omegabh2_lo, omegabh2_hi)`, and the optional `omegam^2 h^2` / `omegamh2` / `omegamh2·n_s` windows). The YAML supplies these through the nested `data.param_cuts:` block; `omegabh2_hi` was the former flat `omegabh2_cut`. Returns `(kept_idx, report)` with a per-window survivor count for the banner; raises on `lo >= hi` or a window whose column (e.g. `ns`) is missing.
- `stream_chunks(idx, chunk)` — yield sorted row-index blocks (sequential disk reads).
- `stream_stats(mm, idx, method, CHUNK)` — per-column mean/std (or min/max) over the used rows, streamed (never loads the dump whole).
- `param_stats(arr, idx, method)` — the same stats for the in-RAM parameter array.
- `read_param_names(covmat_path, comment)` — parameter names from the covmat header line.

### `emulator/geometries_parameter.py` <a name="apx-geometries_parameter"></a>

Input side: raw parameters → whitened network input.

- `ParamGeometry` — center, rotate into the parameter-covariance eigenbasis, unit-scale.
  - `from_covmat` / `from_state` / `state` — build from a covmat file / saved tensors; tensors to save.
  - `whiten` / `unwhiten`, `encode` / `decode` — the transform and its exact inverse.
- `LogParamGeometry` — `ParamGeometry` that whitens in log space for the multiplicative params (`from_samples`, `_to_t` / `_from_t`).
- `AmplitudeFactorGeometry` — whiten every parameter except the IA amplitude(s), append them raw for the loss's closed-form combine (NLA: one amplitude; TATT: three).

### `emulator/geometries_output.py` <a name="apx-geometries_output"></a>

Output side: raw dv ↔ whitened masked target; holds the chi2 covariance. The only file importing cosmolike.

- `DataVectorGeometry` — the base geometry for one probe.
  - `from_cosmolike` / `from_state` / `state` — build from cosmolike / saved tensors; tensors to save.
  - `squeeze` / `unsqueeze` — keep the unmasked entries / scatter them back to full length.
  - `whiten` / `unwhiten`, `encode` / `decode` — covariance-eigenbasis whitening and its inverse.
- `DiagonalGeometry` — whiten by the marginal sigma only (theta order kept, for a CNN).
- `BlockDiagonalGeometry` — whiten each tomographic bin by its own sub-block.
- `build_shear_angle_map(geom, ...)` — attach per-element theta / source-z / xi± branch / per-bin sizes.

### `emulator/analytics.py` <a name="apx-analytics"></a>

Analytic xi rescaling `R` (Eisenstein-Hu zero-baryon preprocessor).

- `_analytic_R(...)` — the formula (numpy or torch); divides out the broadband cosmology dependence.
- `analytic_shape_ratio(...)` — `R` over the masked data vector (the emulator path).
- `rescale_xi(...)` — `R` over the (theta, xi+, xi−) matrix layout (plotting / visual checks).

### `emulator/activations.py` <a name="apx-activations"></a>

Learnable activations for the ResBlock `act` slot.

- `activation_fcn` — the paper's `H` (a learnable identity↔Swish interpolation).
- `GatedActivation` / `PowerGatedActivation` / `GatedPowerActivation` — generalizations (more gates, a bounded power tail, both).
- `make_activation(name, n_gates)` — map a name (`H` / `power` / `multigate` / `gated_power` learnable, plus the parameter-free `relu` / `tanh`) to a factory `act(dim) -> module`.

### `emulator/emulator_designs_building_blocks.py` <a name="apx-building_blocks"></a>

The small `nn.Module`s the models are assembled from.

- `Affine` — a learnable scalar scale + shift (the default ResBlock norm).
- `FeatureAffine` — the per-feature sibling of `Affine`: a length-width gain / bias (one pair per feature; `model.norm per_feature`).
- `make_norm(name)` — map `model.norm` (`affine` / `per_feature` / `none`) to a ResBlock norm factory `norm(size) -> module`; batchnorm deliberately not offered.
- `ResBlock` — width-preserving residual block (n dense layers, each with a norm + activation factory, pre-activation skip).
- `BinLinear` — G per-token *unique* linear layers as one batched einsum; the unique weights also replace the positional encoding.
- `TRFBlock` — one pre-LN transformer block over tokens at their *natural* width (the padded bin length — no embedding/output adapters): shared-weight attention across tokens + a per-token unique MLP stack (the deviation from the textbook shared FFN). The MLP is `n_mlp_blocks` deep and every layer runs at the token width — the interior is pinned to the bin length by design, no width knob. Exactly the identity at init (zero-initialized branch outputs), so a stack satisfies `blocks(x) == x`.
- `FiLMGenerator` — per-channel `gamma` / `beta` produced from the non-amplitude parameters, an identity-init FiLM conditioning of a correction head (amplitude-blind, so the factored exactness holds).
- `rescale_kernel_size` — pick an odd conv kernel width scaled to the bin length.

### `emulator/emulator_designs.py` <a name="apx-emulator_designs"></a>

The full networks.

- `ResMLP` — input projection → residual blocks → output projection → Affine.
- `ResCNN` — ResMLP trunk + a gated bins-as-channels 1D-CNN correction in theta order (one `Conv1d(n_bins → n_bins, k)` kernel over the padded per-bin layout — theta-local and cross-bin, no channel expansion), via fixed basis-change buffers `W_fd` / `W_df` and the `pad_idx` scatter/gather. Head knobs (YAML `model.cnn`): `kernel_size` (tuned as if one block) + `rescale_kernel` (shrink the per-block kernel with depth at a fixed receptive field), `groups` (physical channel cuts: `2` = xi+ never mixes with xi−; on the factored head `3` = GG/GI/II isolated, `6` = both cuts — validated against the mask, other values error), `separable` (factor each block into a depthwise theta filter + pointwise channel mix — a low-rank factorization of the same conv, ~k/2 fewer weights), `film` (re-inject the non-amplitude parameters into every block as an identity-initialized per-channel affine — the head becomes cosmology-aware instead of one fixed map; see `notes/film-conditioning.md`), `n_blocks`, `gate_init`, `activation` (the head's own `{type, n_gates}` family; absent = share the trunk's `model.activation`; a pin needs a frozen-trunk head phase, and `head: activation:` is its alias):

```
  params ─▶ ResMLP trunk ─▶ y    (full-whitened, well-conditioned)
                            │     y @ W_fd   full basis ─▶ theta order
                            ▼
                       1D-CNN blocks         fix theta-local structure
                            │     h @ W_df   theta order ─▶ full basis
                            ▼
              y + gate · correction   ─▶   whitened data vector
```

- `ResTRF` — ResMLP trunk + a gated bin-token transformer correction: the theta-order dv splits into its (xi+/-, source-pair) bins — one bin is one source-redshift-bin pair of xi+ or xi- — (`pad_idx` scatter/gather to a padded per-bin layout, `bin_sizes` from `build_shear_angle_map`), each bin is one token at its natural width (the per-token `n_mlp_blocks`-deep MLPs run at that token width too — no width knob), `TRFBlock`s attend across bins, and the correction is `blocks(h) − h` — zero at epoch 1 because every block starts as the identity. No embedding or output layers (the sequence structure is physical, unlike the published CMB design's latent sequence). Head knobs (YAML `model.trf`): `n_heads`, `n_blocks`, `n_mlp_blocks`, `shared_mlp`, `film`, `gate_init`, and `activation` (the head's own `{type, n_gates}` family; absent = share the trunk's `model.activation`; a pin needs a frozen-trunk head phase, `head: activation:` its alias).

### `emulator/loss_functions.py` <a name="apx-loss_functions"></a>

chi2 losses; each holds a geometry (composition).

- `anneal_value(epoch, opts)` — the per-epoch schedule shared by four knobs: trim, focus, the berhu sqrt-blend, and the EMA horizon.
- `CosmolikeChi2` — the plain chi2: `chi2` ([Mahalanobis](../README.md#14-appendix-the-chi2-metric-mahalanobis) distance), `loss` (trim / focus, and the `chi2` / `sqrt` / `sqrt_dchi2` / `berhu` / `berhu_capped` transform ladder), and thin delegation to the held geometry.
- `berhu` / `berhu_capped` — the reversed-Huber loss modes, configured by a YAML `berhu:` `{knot, cap, anneal}` block: `sqrt(chi2)` below the `knot` chi2, chi2-like above it, and (for `berhu_capped`) sqrt-shaped again past the `cap` so a monster sample's gradient vote is bounded; the optional `anneal:` schedule blends `sqrt` → `berhu` over the run. This is textbook BerHu in the whitened residual norm with delta = sqrt(`knot`), applied per sample as the Mahalanobis aggregate — so `knot` / `cap` are in chi2 units, not residual units.
- `RescaledChi2` — analytic-R "A" form (R divides the net output); `configure_rescaling`, `_R`, `encode` / `decode` / `chi2` / `loss`.
- `ResidualBaseChi2` — analytic-R "B" form (R moves only the baseline; the chi2 stays plain).
- `ElementWeightedChi2` — a per-element focal weight in the training loss (`set_elem_weight`).
- `make_chi2(geom, rescale, ...)` — build the right loss from a geometry and a rescale mode.

### `emulator/batching.py` <a name="apx-batching"></a>

Memory sizing and the regime-aware data loaders. Where the data lives:

```
  dv dump (.npy on disk, memmapped)             never loaded whole
        │   load_source  ─▶  the N_train subset
        ▼
  build_loaders picks a regime by what fits the VRAM budget:
        ├─▶ regime 1   resident on the GPU          encode once; a batch is an on-device index
        ├─▶ regime 2   streamed from host RAM        re-encode each chunk, every epoch
        └─▶ regime 3   streamed from the disk memmap  same, read from disk
```

- `compute_batch_size_bytes` / `compute_model_size_bytes` / `batches_per_load` — per-batch and resident memory estimates.
- `_build_loaders_one(...)` — pick a regime for one source (GPU-resident / RAM-stream / memmap-stream); return `load_C`, `load_dv`, the chunk size, and the bytes it made resident.
- `build_loaders(...)` — run it once per source (train, then val against the reduced budget); return the data dict the loop consumes.

### `emulator/training.py` <a name="apx-training"></a>

The run layer that ties everything together.

- `pick_device` / `make_logger` — setup helpers.
- `make_model` / `make_optimizer` / `make_scheduler` — build one component from a `{cls, **kwargs}` spec dict.
- `build_run_specs(...)` — config → the six `run_emulator` spec dicts.
- `validate_phase_block` / `validate_loss` / `validate_berhu` / `validate_ema` — the pure config validators: the eight-key phase whitelist, the nested `loss:` `{mode, berhu}` block, the berhu `{knot, cap, anneal}` schedule, and the `ema:` `{horizon_epochs, anneal}` block, each checked and canonicalized before the run.
- `derive_eval_bs` / `derive_ema_beta` — turn run-global targets into the derived evaluation batch size (a ~1024-row target) and the per-epoch EMA decay from the horizon.
- `default_train_args` / `suggest_train_args` / `search_defaults` (+ `_as_search_range`, `_range_default`, `_suggest_range`, `_walk_train_args`) — the `[default, min, max, kind]` search resolvers.
- `eval_val` / `eval_source_chi2` — score the model on the val set / per-cosmology delta-chi2.
- `training_loop_batched(...)` — the per-epoch loop (trim / focus / berhu-blend / EMA annealing, best-epoch tracking; an optional Polyak weight average coupled to the best snapshot / rewind).
- `run_emulator(...)` — top-level: build model + optimizer + scheduler + loaders, train, return the histories.
- `audit_devices(model, lossfn, device)` — name every tensor that should live on `device` but does not (a placement check for the compiled forward + loss).

### `emulator/experiment.py` <a name="apx-experiment"></a>

`EmulatorExperiment`: the whole setup as one reusable object.

- `from_yaml` / `from_config` — build from a YAML file / an already-parsed dict.
- `validate_param_cuts` / `validate_sizes` — check the `data` block: the physical-window keys and the absolute `n_train` / `n_val` row counts.
- `resolve_phase_args` / `validate_sweep_paths` — resolve the two-phase keys against the model's real capability (the single-phase demotion) and concretize a sweep's dotted path against that resolved schema.
- `_head_activation_spec` / `_resolve_head_activation` / `_activation_flag_notice` / `_pinned_head_warning` — the per-head activation config layer: validate a `{type, n_gates}` pin, resolve the canonical-vs-alias spelling + the frozen-trunk-head-phase license, and build the flag-vs-pin (from_config) and sweep-vs-pin (bake-off / sweep) startup warnings.
- `stage_train` / `stage_val` / `pool_size` — stage the sources; the physical-cut pool size (the sweep's top N).
- `build_geometry` / `build_specs` — the input/output geometry + chi2; the `run_emulator` spec dicts.
- `train` / `run` — train on the staged data; the full stage→build→train pipeline in one call.
- `frac_above(threshold, ...)` — the sweep metric (fraction of points with delta-chi2 over a cutoff).
- `print_design()` — the shared startup banner: renders the resolved, consumed view (phases resolved to the model's real capability, the model describing itself via `describe_spec`) — device, model class, spec sub-blocks, physical cuts — before anything trains, so a stale YAML is caught at launch.

### `emulator/scheduling.py` <a name="apx-scheduling"></a>

- `lpt_assign(sizes, n_workers)` — split sweep jobs across GPUs by total cost (Longest-Processing-Time).
- `even_assign(jobs, n_workers)` — round-robin split for equal-cost jobs.
- `run_gpu_pool(setup_fn, job_fn, buckets, extra, lanes_per_gpu, job_tokens, on_result)` — the spawned worker pool: one process per (GPU, lane), per-GPU job queues, token gate under packing, parent-side result drain.
- `estimate_train_vram_fraction(n_rows, dv_width, total_bytes)` — conservative per-training VRAM share (`--gpu-pack`).
- `vram_tokens(fraction)` — the packing rule: ≤20% → 1 token, ≤40% → 2, else 4 (exclusive) of `GPU_TOKENS = 4`.

### `emulator/results.py` <a name="apx-results"></a>

- `save_learning_curves(path, sizes, curves, meta)` — write a `np.loadtxt`-friendly plain-text table.
- `save_sweep_table(path, param, values, fracs, meta)` — the one-knob sweep table (numeric values as a column; categorical as an index + label map).
- `save_emulator(path_root, model, param_geometry, geometry, config, histories, train_args, attrs)` — persist a trained run: `.emul` (cpu state_dict, compile prefix stripped) + `.h5` (geometry `state()` groups, per-epoch histories, config YAML, run-identity attrs).

### `emulator/plotting.py` <a name="apx-plotting"></a>

Figures (colorblind-safe palette, no red/green).

- `plot_history` / `plot_learning_curves` / `plot_sweep_curve` / `plot_diagnostics` — the public figures (training history; learning-curve overlay; one-knob hyperparameter-sweep curve; the multipage diagnostics PDF).
- `plot_xi` / `dv_to_xi` / `source_param_samples` — xi correlation-function curves, the dv→matrix reshape, and the coverage-triangle samples.
- `_history_panels` / `_coverage_panels` / `_floor_panel` / `_hard_direction_panels` / `_finish` / `_save_pages` — the shared panel and save helpers.

### `emulator/diagnostics.py` <a name="apx-diagnostics"></a>

Post-training analyses (each returns a dict the plotting reads).

- `coverage_diagnostic(...)` — do the failing val points sit in sparse training regions? (kNN distance vs delta-chi2).
- `local_linear_floor(...)` — the model vs a local-linear interpolation of the data (the data-only floor; plain chi2 only).
- `hard_direction_regression(...)` — which log-parameter combination predicts the per-point hardness.

### `emulator/PCE/` <a name="apx-pce"></a>

NPCE: a sparse-Legendre polynomial-chaos base plus a neural refiner, wired to
the top-level `pce:` YAML block (the base is fit at staging; the refiner is any
`model.name`). `PCEEmulator` deliberately stays out of `MODELS` and
`DesignSpec`: it is loss-owned (wrapped by the PCE losses), not an SGD
architecture.

- `emulator_designs.py` — `PCEEmulator` (the closed-form base; `state()` / `from_state` persist its six buffers to the `.h5` pce group, so inference rebuilds the base with no refit and no cosmolike) + `pce_multi_index`, `pce_design`, `select_lars_loo`.
- `loss_functions.py` — `PCEResidualChi2` (refine the residual), `PCERatioChi2` (refine the ratio) of a frozen PCE base.

### `emulator/IA/` <a name="apx-ia"></a>

Factored intrinsic alignment: emulate cosmology-only templates, apply the IA-amplitude polynomial in closed form.

- `emulator_designs.py` — `TemplateMLP`, `TemplateResCNN`, `TemplateResTRF` (emit the templates, plus optional conv / transformer correction heads).
- `loss_functions.py` — `nla_coeffs`, `tatt_coeffs` (the amplitude polynomials), `NLAAmpFactoredChi2`, `TemplateFactoredChi2` (apply the polynomial in the loss).

### drivers (beside `emulator/`) <a name="apx-drivers"></a>

Each `main()` reads `--root` / `--fileroot` / `--yaml`; the sweep / bake-off add
per-GPU workers.

- `train_single_emulator_cosmic_shear.py` — `main`: one training run + the diagnostics PDF.
- `tune_single_emulator_cosmic_shear.py` — `main` + `_tune_worker` + `journal_storage`: an Optuna study over the YAML's search ranges; serial in-memory, or one worker per GPU sharing a journal-file study.
- `sweep_ntrain_emulator_cosmic_shear.py` — `main` + `_sweep_setup` / `_sweep_job` + `_run_parallel` (LPT split through `run_gpu_pool`) / the serial path; `f(dchi2>thr)` vs `N_train`; `--gpu-pack`.
- `sweep_hyperparam_emulator_cosmic_shear.py` — `main` + `set_by_path` / `read_sweep_block` + `_hyper_setup` / `_hyper_job`; one YAML-chosen knob (`sweep:` block), even split through `run_gpu_pool`; `--gpu-pack`.
- `bakeoff_activation_emulator_cosmic_shear.py` — `main` + `_bakeoff_worker` + `_run_parallel_bakeoff` (activation split) / the serial path; one curve per activation.

---

[← back to the main README](../README.md)
