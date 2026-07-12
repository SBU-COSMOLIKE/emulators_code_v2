**Warning** This pipeline is still in alpha stage `v0.05` and not ready for production. 

# Cosmic-shear data-vector emulator

A neural emulator that maps cosmological parameters to the masked cosmic-shear
(`xi`) data vector, trained against the full-3x2pt chi2 from cosmolike.

Cosmological inference calls a physics pipeline millions of times; this package
trains a neural network to stand in for it, fast enough to run inside the
inference loop. `xi` is the cosmic-shear two-point correlation functions — the
data the analysis measures; cosmolike (inside Cocoa) supplies the analysis mask
and covariance. Accuracy is judged as chi2 — the prediction error in the
covariance units inference actually cares about (the
[chi2 metric](#20-appendix-the-chi2-metric-mahalanobis)).

The pipeline at a glance — every stage is one section of this README:

```
raw dumps            the data-generation run's files: parameters (.txt,
     │               one row per cosmology) + data vectors (.npy)
     ▼  stage
training subset      apply the physical cuts, keep n_train rows
     │               (section 3)
     ▼  whiten
inputs + targets     parameters and data vectors each rescaled so every
     │               direction weighs equally in the fit (section 2)
     ▼  model
whitened prediction  ResMLP, ResCNN, or ResTRF — the network predicts
     │               the whitened data vector (section 10)
     ▼  chi2 loss
per-sample error     each prediction scored by the survey's own error
     │               metric (sections 5 and 15)
     ▼  train
.emul + .h5          the best-epoch weights and everything needed to
                     reload them (section 1)
```

`EmulatorExperiment` wires these stages together. Each driver then varies
exactly one thing: one training run, a hyperparameter search, a
training-set-size sweep, or an activation bake-off ([Run it](#1-run-it)).

The code map — how the package is laid out, what each file does, and where to
edit for a given change — lives in [`emulator/README.md`](emulator/README.md).

## Contents

- **Code map** (layout, file roles, change X → edit Y): [`emulator/README.md`](emulator/README.md)

1. [Run it](#1-run-it)
    1. [Setup: where it runs, and the three path flags](#setup-where-it-runs-and-the-three-path-flags)
    2. [One training run](#one-training-run)
    3. [Run the saved emulator in a Cobaya MCMC](#run-the-saved-emulator-in-a-cobaya-mcmc)
    4. [The `N_train` learning curve](#the-n_train-learning-curve)
    5. [A one-knob sweep](#a-one-knob-sweep)
    6. [A hyperparameter search](#a-hyperparameter-search)
    7. [The activation bake-off](#the-activation-bake-off)
    8. [Packing runs on one big card](#packing-runs-on-one-big-card)
    9. [Where next](#where-next)
    10. [The `sweep:` block (one-knob sweeps)](#sweep-block)
    11. [Multi-GPU execution and packing](#multi-gpu)
    12. [The drivers, family by family](#drivers-table)
2. [The YAML file](#2-the-yaml-file)
3. [`data`](#3-data)
    1. [`param_cuts`](#param_cuts)
4. [Training globals](#4-training-globals)
5. [`loss`](#5-loss)
6. [optimizer, lr, scheduler](#6-optimizer-lr-scheduler)
7. [`trim`](#7-trim)
8. [`focus`](#8-focus)
9. [`ema`](#9-ema)
10. [`model`](#10-model)
    1. [The dense layer](#the-dense-layer)
    2. [`mlp`](#mlp)
    3. [`activation`](#activation)
    4. [`norm`](#norm)
    5. [`cnn` (name `rescnn`)](#cnn-name-rescnn)
    6. [`trf` (name `restrf`)](#trf-name-restrf)
11. [Two-phase schedule + the `trunk:` / `head:` blocks](#11-two-phase-schedule--the-trunk--head-blocks)
12. [`pce`](#12-pce)
    1. [What the base is made of](#what-the-base-is-made-of)
    2. [One term = a product of one-parameter curves](#one-term--a-product-of-one-parameter-curves)
    3. [Which terms are allowed](#which-terms-are-allowed)
    4. [What gets fit, mode by mode](#what-gets-fit-mode-by-mode)
    5. [The combine form and the block](#the-combine-form-and-the-block)
13. [Starting from a saved emulator: fine-tuning + transfer](#13-starting-from-a-saved-emulator-fine-tuning--transfer)
    1. [Fine-tuning (`train_args.finetune`)](#fine-tuning-train_argsfinetune)
    2. [Transfer learning (`transfer:`)](#transfer-learning-transfer)
    3. [Joint refinement (`transfer.refine`, optional stage 2)](#joint-refinement-transferrefine-optional-stage-2)
14. [Scalar (derived-parameter) emulators](#14-scalar-derived-parameter-emulators)
15. [Emulating CMB spectra (TT / TE / EE / phi-phi)](#15-emulating-cmb-spectra-tt--te--ee--phi-phi)
16. [Emulating the expansion history (H(z), BAO and SN distances)](#16-emulating-the-expansion-history-hz-bao-and-sn-distances)
17. [Emulating the matter power spectrum (hybrid inference, EMUL2)](#17-emulating-the-matter-power-spectrum-hybrid-inference-emul2)
18. [Generating the training set](#18-generating-the-training-set)
19. [Appendix: the pipeline](#19-appendix-the-pipeline)
20. [Appendix: the chi2 metric (Mahalanobis)](#20-appendix-the-chi2-metric-mahalanobis)
21. [Appendix: activation functions](#21-appendix-activation-functions)
    1. [The paper's $H(x)$](#the-papers-hx)
    2. [Generalizations](#generalizations)
    3. [Selecting one](#selecting-one)
22. [Appendix: precedence — who wins when settings collide](#22-appendix-precedence--who-wins-when-settings-collide)
    1. [A. Activation family](#a-activation-family)
    2. [B. Phase blocks vs the top level (two-phase models)](#b-phase-blocks-vs-the-top-level-two-phase-models)
    3. [C. Single-phase demotion (the same YAML on `resmlp`)](#c-single-phase-demotion-the-same-yaml-on-resmlp)
    4. [C2. Two-phase schedule modes](#c2-two-phase-schedule-modes)
    5. [D. Loss-block spellings](#d-loss-block-spellings)
    6. [E. Sweeps and searches](#e-sweeps-and-searches)
    7. [F. Constructor / driver args vs the YAML](#f-constructor--driver-args-vs-the-yaml)
    8. [G. Deliberately no knob (nothing to win)](#g-deliberately-no-knob-nothing-to-win)
23. [Appendix: scripting a saved emulator (without Cobaya)](#23-appendix-scripting-a-saved-emulator-without-cobaya)
24. [AI-Usage](#24-ai-usage)

---

## 1. Run it

### Setup: where it runs, and the three path flags

Training needs a machine with a working Cocoa installation — cosmolike
supplies the data-vector mask and covariance — and, in practice, a CUDA GPU;
the `emulator/` package itself is pure PyTorch and can be read or developed
anywhere. Every driver reads the same three path flags, so set the driver
folder once:

```bash
# run from $ROOTDIR (cocoa exports it). --root = project folder under $ROOTDIR;
# --fileroot = a subfolder of it holding this emulator's YAML + outputs; --yaml =
# a bare filename under --fileroot. Data (dv/params/covmat) lives in
# --root/chains, the project's data folder.
D=external_modules/code/emulators/emultrfv2
```

### One training run

Train the YAML's model once; `--diagnostic` adds a
multipage PDF of accuracy diagnostics:

```bash
python $D/cosmic_shear_train_emulator.py \
  --root projects/lsst_y1/ --fileroot emulators/training_scripts/ \
  --yaml cosmic_shear_train_emulator.yaml --diagnostic diagnostic
```

This writes the trained emulator under `--root/chains` as a `.emul` / `.h5`
pair: the `.emul` holds the best-epoch weights, and the `.h5` carries both
whitening geometries, the per-epoch histories, and the fully-resolved config
(schema v2) — so a saved emulator rebuilds bit-exactly even if code defaults
later change (`rebuild_emulator` in `emulator/results.py` reads the file
alone).

### Run the saved emulator in a Cobaya MCMC

Once saved, the emulator runs inside a Cobaya MCMC through the thin Theory
adapter `cobaya_theory/emul_cosmic_shear.py`: point its `emulators:` list at
the saved path root and it rebuilds the module and reads the sampled
parameters it needs straight from the `.h5` — no architecture, no parameter
ordering re-typed in the sampling YAML (the legacy `ord` / `extrapar` are
retired). The whole theory block is:

```yaml
theory:
  emul_cosmic_shear:
    # python_path (NOT path) is cobaya's external-class key; without it a
    # legacy adapter bundled in cocoa's cobaya fork shadows this class.
    python_path: ./external_modules/code/emulators_code_v2/cobaya_theory/
    stop_at_error: True
    extra_args:
      device: 'cuda'
      # one path root per emulator -> <root>.h5 + <root>.emul,
      # ROOTDIR-relative; everything else is read from the .h5.
      emulators:
        - projects/lsst_y1/chains/emulator_resmlp_t256_ntrain250000
```

The adapter itself holds no physics. It reads the sampled parameter names
from the `.h5`, asks cobaya for them at every step, and hands them to the
predictor:

```
sampling YAML         the theory block above: device + the emulators list
     │
     ▼  cobaya_theory/emul_cosmic_shear.py
thin adapter          no physics here — it only moves parameters in and
     │                the data vector out
     ▼  emulator/inference.py  (EmulatorPredictor)
encode                rescale the sampled parameters into the units the
                      network was trained on
forward               the rebuilt network predicts the data vector in its
                      trained-on, rescaled units
combine               the factored-IA or NPCE recombination, when the
                      artifact was trained with one
decode                undo the rescaling: back to the physical data vector
     │
     ▼
likelihood            scores it as usual
```

A copyable full evaluate config — the likelihood, params, and sampler
blocks wrapped around this theory block — ships as
`cobaya_theory/EXAMPLE_EMUL_EVALUATE.yaml`.

An emulator can also replace CAMB *inside* cosmolike rather than
replacing cosmolike itself: cosmolike's `use_emulator: 2` mode consumes
emulated CAMB products — the matter power spectrum, the expansion
history, and the sound horizon r_drag — served by `emul_mps`,
`emul_baosn`, and `emul_scalars` as three theory blocks in one sampling
YAML. That hybrid pattern is
[section 17](#17-emulating-the-matter-power-spectrum-hybrid-inference-emul2).

### The `N_train` learning curve

How does accuracy improve as the training set grows? This driver retrains the same model at several training-set sizes and
plots the error metric (`frac>0.2`, defined in [section 2](#2-the-yaml-file))
against N: a curve still falling at the largest N says more data will help; a
flat tail says the model, not the data, is the limit. Sweep points are
independent trainings, so they run in parallel — one whole training per GPU,
all visible GPUs by default ([Multi-GPU](#multi-gpu) below):

```bash
python $D/cosmic_shear_sweep_ntrain_emulator.py \
  --root projects/lsst_y1/ --fileroot emulators/training_scripts/ \
  --yaml cosmic_shear_train_emulator.yaml --n-points 8 --out curve
```

### A one-knob sweep

Pick one knob — a learning rate, a kernel size, a batch size — and train
the full model once per value, so that knob's effect shows up in isolation.
The knob and its values live in a top-level `sweep:` block, named by the
knob's dotted path:

```yaml
sweep:
  parameter: lr.lr_base
  values:
    - 0.0010
    - 0.0025
    - 0.0063
```

The block's full rules are in [The `sweep:` block](#sweep-block). Run it
with:

```bash
python $D/cosmic_shear_sweep_hyperparam_emulator.py \
  --root projects/lsst_y1/ --fileroot emulators/training_scripts/ \
  --yaml cosmic_shear_train_emulator.yaml --out lrsweep
```

### A hyperparameter search

Optuna (https://optuna.org) is a search library: it
proposes trial settings, watches the results, and concentrates new trials
where the metric improves. The searched ranges are the YAML's
`[default, min, max, kind]` leaves — any numeric leaf may swap its scalar
for a 4-list, and only those leaves are searched (every other driver
collapses them back to the default):

```yaml
train_args:
  bs: 256                                # fixed scalar: never searched
  lr:
    lr_base:       [2.5e-3, 1.0e-4, 1.0e-2, log]   # [default, min, max, kind]
    warmup_epochs: [10, 0, 30, int]
  model:
    mlp:
      width:    [128, 64, 256, int]      # kind = int | float | log
      n_blocks: [4, 2, 6, int]
```

`--n-trials` bounds the study:

```bash
python $D/cosmic_shear_tune_emulator.py \
  --root projects/lsst_y1/ --fileroot emulators/training_scripts/ \
  --yaml cosmic_shear_tune_emulator.yaml --n-trials 64
```

### The activation bake-off

Trains the same model once per activation family
over a grid of training sizes and overlays their learning curves: a
head-to-head showing whether a family genuinely learns faster (a lower curve
everywhere) or just ties:

```bash
python $D/cosmic_shear_bakeoff_activation_emulator.py \
  --root projects/lsst_y1/ --fileroot emulators/training_scripts/ \
  --yaml cosmic_shear_train_emulator.yaml --out bakeoff
```

### Packing runs on one big card

On a card with far more memory than one training needs, such as an H200, add
`--gpu-pack` to either sweep: points estimated at ≤ 20% of the GPU run four
to a card, ≤ 40% two to a card, bigger ones exclusive (off by default — on a
12 GB RTX 3060 one training is the card). The details live in
[Multi-GPU execution and packing](#multi-gpu) below.

### Where next

The YAML has two top-level blocks:

```yaml
data:          # where the training vectors come from, how many rows to use
  ...
train_args:    # the whole run: objective, optimizer, schedules, model
  ...
```

The next
chapter ([The YAML file](#2-the-yaml-file), sections 2–11) documents every
block with its math, options, and a small example; templates live in
`example_yamls/` (one per driver style — copy one into your `--fileroot` and
edit it). The `sweep:` block is documented [below](#sweep-block).

### The `sweep:` block (one-knob sweeps) <a name="sweep-block"></a>

`cosmic_shear_sweep_hyperparam_emulator.py` (and the per-family
`<family>_sweep_hyperparam_emulator.py` siblings, which run the same
sweep serially) reads one extra top-level YAML
block (the other drivers ignore it) naming exactly one `train_args` leaf by
its dotted path, and the values to try — one full training per value at
fixed `N_train`:

```yaml
sweep:
  parameter: lr.lr_base
  values:
    - 0.0010
    - 0.0025
    - 0.0063
```

| Rule | Why |
|---|---|
| any `train_args` leaf sweeps by dotted path (`bs`, `trim.start`, `model.cnn.kernel_size`, `model.cnn.film`, `head.lr.lr_base`, …) | the sweep deep-copies `train_args` and sets that one leaf per point |
| `model.activation` (or `.type`) is a special case | the activation family is resolved onto the experiment at build, not read from `train_args`; the driver sets it per value — leave `--activation` unset |
| `model.name` / `model.ia` are refused | they change the model *class*; run one sweep per architecture and overlay the tables |
| an unknown first segment is refused | a typo'd path would otherwise silently train the same config N times |
| a missing intermediate block is created (`head.lr.lr_base` with no `head:` block) | but a phase axis (`head.*` / `trunk_epochs` / `trunk.*`) on a single-phase model is rejected up front by `validate_sweep_paths` (it would be demoted away) |

The sweep writes two files under `--fileroot`: a results table
(`<--out>.txt`) and the same results drawn as a curve (`<--out>.pdf`).
Each table row is one swept value and the error metric it reached. When the
swept knob is a number (a batch size, a learning rate), the first column is
the value itself:

```text
# sweep: f(delta-chi2 > threshold) vs lr.lr_base
# model=rescnn  threshold=0.2  n_train=250000
# columns: lr.lr_base, frac
0.001  0.401234
...
```

When the knob is a word or a switch (an activation name, film on/off), the
first column is an integer index instead, and a comment line at the top says
which index means which setting:

```text
# values: 0=H, 1=power, 2=multigate
# columns: index, frac
0  0.401234
...
```

Both layouts load with a plain `np.loadtxt` (the labels live in comment
lines). The full template is
`example_yamls/cosmic_shear_sweep_hyperparam_emulator.yaml`, with the common
sweeps (bs, activation family, film on/off, conv depth, head lr) ready to
swap in.

### Multi-GPU execution and packing <a name="multi-gpu"></a>

Every hyperparameter driver uses all visible CUDA devices by default
(`--n-gpus` caps it; one GPU or Apple MPS runs serially). Jobs never split
across GPUs — one whole training per spawned worker:

| Driver | Jobs | Split across GPUs | Extra flags |
|---|---|---|---|
| `sweep_ntrain` | one training per `N_train` | LPT (cost ∝ N: biggest first to the least-loaded GPU) | `--gpu-pack` |
| `sweep_hyperparam` | one training per value | round-robin (equal cost) | `--gpu-pack` |
| `bakeoff_activation` | one learning curve per activation | by activation | |
| `tune` | Optuna trials | one worker per GPU, one shared study | `--journal` |

**`--gpu-pack` (both sweep drivers; off by default)** runs several small
trainings on the same card at once. A small model often leaves the card
mostly idle — the wall time goes into Python dispatching work, not into the
card computing — so sharing the card finishes the sweep faster. How many
runs share: each run's GPU-memory need is estimated up front, and a run
expected to fit in 20% of the card shares it with up to three others, one
fitting in 40% shares with one other, and anything bigger gets the card to
itself. Turn it on for big cards with small-to-mid `N_train`; leave it off
on small cards, or whenever per-epoch timings must stay comparable across
runs (sharing makes each run slower and noisier). The memory estimate and
the sharing arithmetic live in `scheduling.py`
(`estimate_train_vram_fraction`, `vram_tokens`).

**Parallel Optuna (`cosmic_shear_tune_emulator.py --n-gpus N`)** shares a single study through a
journal file (`--journal`): the parent enqueues the warm-start, `--n-trials`
splits across workers, and reusing the journal resumes it (serial on 1 GPU/MPS).

### The drivers, family by family <a name="drivers-table"></a>

The driver namespace is `<verb>_<family>_emulator.py`. Every family trains
through the same config dispatch — the `data` block names the family
(`data.outputs` -> scalar, `data.cmb` -> CMB, `data.grid` -> background,
`data.grid2d` -> matter power; a cosmolike block -> cosmic shear) — so the
per-family drivers differ only in their prog names and defaults.

| Driver | Family | What it does |
|---|---|---|
| cosmic_shear_train_emulator.py | cosmic shear (any data-block family) | train one emulator from a YAML — the family comes from the data block; also the shared engine the three family train drivers wrap; `--diagnostic` writes the multipage PDF |
| scalar_train_emulator.py | scalar | train one derived-parameter emulator; `--diagnostic` |
| cmb_train_emulator.py | cmb | train one CMB-spectrum emulator (requires a data.cmb block, wrong-family YAMLs name the right driver); `--diagnostic` adds the CMB pages |
| baosn_train_emulator.py | baosn | train one background emulator (requires data.grid); `--diagnostic` adds the redshift + derived-distance pages |
| mps_train_emulator.py | mps | train one matter-power emulator (requires data.grid2d); `--diagnostic` adds the (z, k) residual-surface pages |
| cosmic_shear_sweep_ntrain_emulator.py | cosmic shear | f(delta-chi2 > thr) vs `N_train`, multi-GPU pool + gpu-pack |
| scalar_sweep_ntrain_emulator.py | scalar | the same learning curve (thin wrapper: multi-GPU + `--gpu-pack` carry over) |
| cmb_sweep_ntrain_emulator.py | cmb | the same (thin wrapper) |
| baosn_sweep_ntrain_emulator.py | baosn | the same (thin wrapper) |
| mps_sweep_ntrain_emulator.py | mps | the same (thin wrapper) |
| cosmic_shear_tune_emulator.py | cosmic shear | Optuna study, multi-GPU journal |
| scalar_tune_emulator.py | scalar | Optuna study (thin wrapper: serial or `--n-gpus` journal study) |
| cmb_tune_emulator.py | cmb | the same |
| baosn_tune_emulator.py | baosn | the same |
| mps_tune_emulator.py | mps | the same |
| cosmic_shear_bakeoff_activation_emulator.py | cosmic shear | activation bake-off learning curves |
| cosmic_shear_sweep_hyperparam_emulator.py | cosmic shear | one-axis hyperparameter sweeps, multi-GPU |
| {scalar,cmb,baosn,mps}_sweep_hyperparam_emulator.py | each family | thin wrappers over the cosmic-shear driver: same one-knob sweep, multi-GPU + `--gpu-pack` |

The four cosmic-shear drivers still carry their original names; renaming
them into the namespace is a recorded polish item that lands after the
first full gate-board run, and the board configs and README references
move with it.

---

## 2. The YAML file

The YAML has two top-level blocks. `data` says where the training vectors
come from and how many rows to use. `train_args` describes the whole run:
objective, optimizer, schedules, model. Any numeric leaf may be a plain
scalar or a `[default, min, max, kind]` search range, where `kind` is `int`,
`float`, or `log`; only the `*_tune_emulator.py` drivers search the
ranges, and every other driver collapses a range to its default value. Sections 3–11 document each
block. When two settings collide, the winner is defined in the
[precedence appendix](#22-appendix-precedence--who-wins-when-settings-collide).
Templates live in `example_yamls/`, and the `sweep:` block is described in
[Run it](#sweep-block).

One compact production run (two-phase `restrf` + `nla`, a berhu head):

```yaml
data:
  train_dv:     w0wa_takahashi_dvs_train_cs_16.npy
  train_params: w0wa_takahashi_params_train_cs_16.1.txt
  train_covmat: w0wa_takahashi_params_train_cs_16.covmat
  val_dv:       w0wa_takahashi_dvs_train_cs_8.npy
  val_params:   w0wa_takahashi_params_train_cs_8.1.txt
  cosmolike_data_dir: lsst_y1
  cosmolike_dataset:  lsst_y1_M1_GGL0.05.dataset
  n_train: 25000
  n_val:   5000
train_args:
  nepochs: 1600
  bs:      256
  loss:
    mode: sqrt
  model:
    name: restrf
    ia:   nla
    mlp:
      width:    128
      n_blocks: 4
```

Six terms the chapter uses. The details live in appendices
[19](#19-appendix-the-pipeline) and
[20](#20-appendix-the-chi2-metric-mahalanobis).

| Term | Meaning |
|---|---|
| data vector, dv | The masked cosmic-shear two-point functions xi+/- stacked into one vector. This is what the network predicts. |
| chi2 | Prediction error measured in the analysis covariance, `r^T Cinv r` — [appendix 20](#20-appendix-the-chi2-metric-mahalanobis). The headline metric is written `frac>0.2` in the logs: the fraction of validation cosmologies with delta-chi2 above 0.2. The goal is to drive it down. |
| whitened | Rotated and rescaled so the components are decorrelated with unit variance. This is the form the network sees, input and output — [appendix 19](#19-appendix-the-pipeline). |
| theta order | The data vector re-sorted to vary smoothly along the angular axis. The correction heads work in this basis. |
| trunk / head | Every architecture is a shared ResMLP trunk; `rescnn` and `restrf` add a gated correction head on top — [section 10](#10-model). |
| dump | The big on-disk table of parameters and data vectors the physics code wrote. Training memmaps it — reads slices from disk, never the whole file — and stages only the rows it needs, [section 3](#3-data). |

---

## 3. `data`

The training and validation vectors, and how many rows to keep. Five bare
filenames resolve under `--root/chains` (three train: `train_dv` /
`train_params` / `train_covmat`; two val: `val_dv` / `val_params`); the
`cosmolike_data_dir` / `cosmolike_dataset` pair instead resolves under
`$ROOTDIR/external_modules/data`. `n_train` and `n_val` are **absolute row
counts** (not fractions), enforced *after* the physical cuts — if the cut pool
holds fewer rows the run raises rather than training on less than you asked.
`split_seed` seeds the shuffle; `ram_frac` is the fraction of free RAM staging
may fill before it streams from the disk memmap instead. Where these dumps
come from — how the training parameters are sampled, whitened, and named — is
[section 18](#18-generating-the-training-set).

```
dv/params dump ─▶ seeded shuffle ─▶ param_cuts ─▶ first n_train (+ n_val)
                                                        │
                          fits ram_frac of free RAM? ───┤
                             yes: resident in RAM   no: streamed from the memmap
```

```yaml
data:
  train_dv:     w0wa_takahashi_dvs_train_cs_16.npy
  train_params: w0wa_takahashi_params_train_cs_16.1.txt
  train_covmat: w0wa_takahashi_params_train_cs_16.covmat
  val_dv:       w0wa_takahashi_dvs_train_cs_8.npy
  val_params:   w0wa_takahashi_params_train_cs_8.1.txt
  cosmolike_data_dir: lsst_y1
  cosmolike_dataset:  lsst_y1_M1_GGL0.05.dataset
  n_train:    25000
  n_val:      5000
  split_seed: 0
  ram_frac:   0.7
```

### `param_cuts`

Physical density windows that keep the training set inside the region the
emulator must be accurate on. Each window is a `_lo` / `_hi` pair; omit a key
for no cut on that side, and `lo >= hi` raises.

| keys | quantity | formula | Planck |
|---|---|---|---|
| `omegabh2_lo/_hi` | $\Omega_b h^2$ | $\Omega_b (H_0/100)^2$ | 0.0224 |
| `omegam2h2_lo/_hi` | $\Omega_m^2 h^2$ | $(\Omega_m H_0/100)^2$ | 0.045 |
| `omegamh2_lo/_hi` | $\Omega_m h^2$ | $\Omega_m (H_0/100)^2$ | 0.143 |
| `omegamh2ns_lo/_hi` | $\Omega_m h^2 n_s$ | $\Omega_m h^2 \cdot n_s$ | 0.138 |

The last needs the $n_s$ column in the params file.

```yaml
  param_cuts:
    omegabh2_hi:  0.035
    omegabh2_lo:  0.014
    omegam2h2_lo: 0.015
    omegam2h2_hi: 0.08
```

---

## 4. Training globals

The run-level knobs that are not their own block:

| Knob | What it does |
|---|---|
| `nepochs` | Passes over the training set. |
| `bs` | The training minibatch size. Validation uses its own batch size, derived to target ~1024 rows by `derive_eval_bs`, so a small `bs` does not slow scoring. |
| `trunk_epochs` / `freeze_trunk` | The two-phase schedule of [section 11](#11-two-phase-schedule--the-trunk--head-blocks). The mode table is precedence [C2](#22-appendix-precedence--who-wins-when-settings-collide). |
| `silent` | Suppress the per-epoch progress lines. |
| `clip` | A per-step ceiling on the gradient norm; `0` turns it off. The whole gradient is rescaled toward the ceiling and keeps its direction, so one batch holding an extreme sample cannot kick the weights. The rule is below. |
| `rewind` | On every learning-rate cut by the plateau scheduler of [section 6](#6-optimizer-lr-scheduler), reload the best weights and optimizer snapshot while keeping the reduced rate. An excursion into a bad basin then costs at most `patience` epochs. |

$$g \leftarrow g \cdot \min\left(1,\ \frac{\mathrm{clip}}{\lVert g \rVert}\right)$$

```yaml
train_args:
  nepochs: 1600
  bs:      256
  silent:  false
  # clip:   1.0
  # rewind: true
```

---

## 5. `loss`

The training objective. `loss.mode` picks a per-sample transform $L(c)$ of
each sample's chi2 $c = r^\top C^{-1} r$
([Mahalanobis](#20-appendix-the-chi2-metric-mahalanobis)); the batch loss is
the (trimmed, focally weighted; [sections 7–8](#7-trim)) mean of $L(c)$. The transform sets how a
sample's gradient vote scales with its misfit:

| mode | $L(c)$ | vote vs misfit | use it when |
|---|---|---|---|
| `chi2` | $c$ | grows with $c$ (tail-chasing) | the fit is already close everywhere |
| `sqrt` | $\sqrt{c}$ | equal for every sample | the default; robust to a fat tail |
| `sqrt_dchi2` | $\sqrt{1+2c}-1$ | equal, softer near 0 | a smoother sqrt |
| `berhu` | reversed Huber (below) | equal in the bulk, rising in the window | push the bulk under the goal |
| `berhu_capped` | berhu, then flat | rising, then bounded above the cap | as berhu, monster-robust |

In closed form, `chi2` is $c$, `sqrt` is $\sqrt{c}$, and `sqrt_dchi2` is
$\sqrt{1+2c}-1$. `berhu` is the reversed Huber,

$$L_{\mathrm{berhu}}(c) = \sqrt{c} \ \ (c \le k), \qquad
\dfrac{c+k}{2\sqrt{k}} \ \ (c > k)$$

and `berhu_capped` adds $\dfrac{2\sqrt{Kc}+k-K}{2\sqrt{k}}$ for $c > K$.

$C^1$ at every knot. $k$ = `berhu.knot` (default 0.2, the frac>0.2 goal),
$K$ = `berhu.cap` (default 10). This is textbook BerHu in the whitened
residual norm with $\delta = \sqrt{k}$ — the knots are in chi2 units, applied
per sample (the Mahalanobis aggregate). Vote intuition: `sqrt` gives every
sample an equal vote; `chi2`'s vote grows with $c$ (the tail dominates);
`berhu` keeps equal bulk votes and rises ~×7 across $(k, K)$; `berhu_capped`
plateaus above $K$ so a chi2=100 monster stays bounded.

The `berhu:` sub-block sets the knots (spell it `berhu:` — the family, so it
survives a `mode` sweep — or after the active mode as `berhu_capped:`; giving
both is an error, see precedence
[D](#22-appendix-precedence--who-wins-when-settings-collide)). An optional
`anneal:` (presence = on) starts as plain sqrt and blends into the berhu shape
on the [shared schedule](#7-trim), $s: 0 \to 1$:

$$L_s = (1-s) \sqrt{c} + s L(c)$$

```yaml
  loss:
    mode: berhu_capped
    berhu:
      knot: 0.2
      cap:  10
      # anneal:
      #   hold_epochs:   50
      #   anneal_epochs: 300
      #   shape:         cosine
```

---

## 6. optimizer, lr, scheduler

The optimization stack — three small blocks that interact, so they are read
together.

| Block | What it does |
|---|---|
| `optimizer` | The class is fixed to **AdamW**. `weight_decay` decays only the true weight matrices — the `.weight` of `Linear`, `Conv1d`, and `BinLinear` — never biases, norms, or activation parameters. On CUDA the faster fused kernel is used. The full decay rule is detailed below. |
| `lr` | The learning rate scales with the square root of the batch size: bigger batches average away gradient noise, so the step can grow. The formula is below. `bs_base` is the run-global anchor and never sits inside a phase block. `warmup_epochs` ramps the rate linearly from 0 over the first epochs. |
| `scheduler` | The class is fixed to **ReduceLROnPlateau**, with `mode`, `patience`, and `factor` as its settings, stepped every epoch on the **raw** validation median — the EMA average never feeds it. A per-phase `scheduler:` replaces the settings but keeps the class; see precedence [B](#22-appendix-precedence--who-wins-when-settings-collide). |

$$\mathrm{lr} = \ell \sqrt{B/B_0}$$

with $\ell$ = `lr_base`, $B$ = `bs`, and $B_0$ = `bs_base`.

```yaml
  optimizer:
    weight_decay: 0.0
  lr:
    lr_base:       0.0025
    bs_base:       64.0
    warmup_epochs: 10
  scheduler:
    mode:     min
    patience: 25
    factor:   0.8
```

**Weight decay — only true weight matrices.** AdamW's decoupled decay adds a
small pull toward 0 beside the gradient step:

$$w \leftarrow w - \mathrm{lr} \lambda w$$

with $\lambda$ = `weight_decay` (AdamW's decoupled decay, applied beside the
gradient step). Membership is decided by module role, not tensor shape:

```
    optimizer.weight_decay: lambda
                 │
     decayed ◀───┴───▶ never decayed
┌──────────────────────┐   ┌─────────────────────────────────┐
│ weight matrices only │   │ everything else:                │
│   Linear.weight      │   │   all biases                    │
│   Conv1d.weight      │   │   Affine / FeatureAffine g, b   │
│   BinLinear.weight   │   │   activation parameters         │
└──────────────────────┘   │   (H's gamma/beta; multigate's  │
                           │   w/beta/mu — any shape)        │
                           └─────────────────────────────────┘
```

Decay limits function complexity by preferring small weight matrices; pulling
biases or activation shape parameters toward 0 has no such meaning — it drags
the activation toward degenerate forms (gates dying, centers collapsing) — so
membership is decided by module role, never by tensor shape, no matter how many
parameters a family carries.

```yaml
  optimizer:
    weight_decay: 1.0e-4   # L2 pull on the weight matrices only
                           # (Linear / Conv1d / BinLinear weights);
                           # biases, Affine / FeatureAffine gains,
                           # and every activation parameter are
                           # never decayed, whatever their shape.
                           # 0 = off (the template default).
```

---

## 7. `trim`

Drop the worst `trim(e)` fraction of each batch before the mean — a hard
reject, so a few contaminated vectors cannot dominate the gradient (eval never
trims). `trim(e)` runs the **shared annealed schedule** (`anneal_value`), the
same machinery `focus`, `ema.anneal`, and `loss.berhu.anneal` reuse with their
own `start` / `end`:

```
value
start ───────────╮
                 │╲     shape: cosine | linear | step | const
                 │ ╲
end   ───────────┴──╲──────────────
        hold          anneal          (epochs)
```

Hold `start` for `hold_epochs`, ramp to `end` over `anneal_epochs`, then stay
at `end`. `cosine` eases with zero slope at both ends (no abrupt jumps to
mislead the plateau scheduler); `linear` is a straight ramp; `step` floors the
linear ramp to a 0.01 grid; `const` holds `start` forever (`end` /
`hold_epochs` / `anneal_epochs` ignored). Advice: keep `end > 0`, a floor that
keeps the very worst fraction out for the whole run.

```yaml
  trim:
    start:         0.1
    end:           0.025
    hold_epochs:   50
    anneal_epochs: 300
    shape:         cosine
```

---

## 8. `focus`

Focal hardness weighting: up-weight the harder samples in the mean by a
detached weight (a sample cannot lower its own weight instead of fitting):

$$w_i = \left(\frac{c_i}{c_i + \kappa}\right)^{\gamma(e)}$$

`start` / `end` / `hold_epochs` / `anneal_epochs` / `shape` run $\gamma(e)$ on
the [shared schedule](#7-trim) (0 = a plain mean early, `end` ≈ 2 typical);
`kappa` is the chi2 scale where a sample's hardness crosses 1/2, fixed over the
run. A negative `gamma` is the off sentinel ($w_i = 1$ everywhere). Interplay:
`berhu` already carries the tail-emphasis role, so soften `focus` on a berhu
head.

```yaml
  focus:
    start:         0.0
    end:           2.0
    hold_epochs:   50
    anneal_epochs: 300
    shape:         linear
    kappa:         0.15
```

---

## 9. `ema`

An optional Polyak weight average (a running average of the network weights;
the averaged copy is what ships), updated after every optimizer step:

$$\bar\theta \leftarrow \beta \bar\theta + (1-\beta) \theta
\qquad \beta = 1 - \frac{1}{H S}$$

with $H$ = `horizon_epochs` and $S$ = steps per epoch. The horizon is set
in **epochs**, so $\beta$ (and the effective window) is
batch-size-invariant. Selection and the reported metrics run on $\bar{\theta}$;
the scheduler stays on the raw median; the `{theta, optimizer, theta_bar}`
triple is snapshotted and rewound as one unit. An optional `anneal:` runs the
[shared schedule](#7-trim) on the horizon, $h(e) = \mathrm{horizon}\cdot
s(e)$, so the average carries no memory of the terrible early era. Absent =
off (byte-identical). A per-phase `ema:` fully replaces this one; `ema: null`
turns it off for that phase.

```yaml
  ema:
    horizon_epochs: 3
    anneal:
      hold_epochs:   50
      anneal_epochs: 300
      shape:         cosine
```

---

## 10. `model`

Two orthogonal choices — `name` (the architecture) and the optional `ia` (the
factored intrinsic-alignment design) — pick one of six classes:

| | plain | `ia: nla` | `ia: tatt` |
|---|---|---|---|
| `resmlp` | `ResMLP` | `TemplateMLP` | `TemplateMLP` |
| `rescnn` | `ResCNN` | `TemplateResCNN` | `TemplateResCNN` |
| `restrf` | `ResTRF` | `TemplateResTRF` | `TemplateResTRF` |

Every architecture is a shared ResMLP trunk; `rescnn` and `restrf` add a
gated correction head in theta order. The gate is one learned number
scaling the correction — small at the start, so every architecture begins
as its trunk:

```
params ─▶ ResMLP trunk ─▶ y, in the rescaled units of section 2
                          │  fixed basis change to theta order
                          ▼
                    head blocks: cnn conv | trf attention
                          │
          y + gate · correction ─▶ data vector (rescaled)
```

Intrinsic alignments (IA): galaxies have correlated intrinsic shapes
independent of lensing — the main astrophysical contaminant of cosmic shear;
the analysis models it with amplitude parameters. An `ia` design makes the
network emit a set of templates — data-vector-shaped outputs in the same
rescaled units — that a closed-form polynomial combines, so the IA
amplitudes never enter the network and the emulator is exact in them. For `nla`
(3 templates, amplitude $A_1$):

$$\xi = K_0 + A_1 K_1 + A_1^2 K_2$$

For `tatt` (10 templates; amplitudes $a_1, a_2, b_{TA}$ — the
IA field is $a_1 O_1 + a_2 O_2 + a_1 b_{TA} O_{1\delta}$, so
$\xi$ is quadratic in it: 1 GG + 3 GI + 6 II terms):

$$\xi = K_0 + a_1 K_1 + a_2 K_2 + a_1 b_{TA} K_3 +
a_1^2 K_4 + a_2^2 K_5 + (a_1 b_{TA})^2 K_6 +
a_1 a_2 K_7 + a_1^2 b_{TA} K_8 + a_1 a_2 b_{TA} K_9$$

### The dense layer

Every learned layer in this library is one primitive repeated: the
**linear layer** — multiply the incoming numbers by a matrix of
learned weights, then add a learned offset.

```
x = (x_1, ..., x_n)              the input: n numbers
        │
        ▼      y_i = W_i1·x_1 + W_i2·x_2 + ... + W_in·x_n + b_i
   ┌──────────┐
   │ W  n × n │   output number i is a weighted sum of all n inputs —
   │ b  n     │   the weights are row i of W — plus an offset b_i.
   └──────────┘   W and b hold the learned numbers, and this weighted
        │         sum is everything a linear layer does.
        ▼
y = (y_1, ..., y_n)              the output: n numbers again
```

A **dense layer** is a linear layer inside a block, and in this
library a dense layer never changes dimension: W is always square,
`width` numbers in, `width` numbers out. Every dense layer everywhere
obeys this — the trunk's residual blocks, the transformer head's
attention matrices, each bin's private stacks. The fixed width is a
design rule, and it pays three times:

| where | what the fixed width buys |
|---|---|
| residual blocks | the skip `+` adds a block's input to its output directly — same length by construction, so no extra layer is spent making the shapes match |
| correction heads | channels and tokens stay at their physical size, so the heads need no adapter layers in or out — the [`cnn`](#cnn-name-rescnn) and [`trf`](#trf-name-restrf) subsections show what that removes |
| reading a config | one number, `model.mlp.width`, describes every dense layer in the trunk |

The dimension still has to change somewhere: a dozen parameters enter,
a data vector of hundreds of numbers leaves. That happens in exactly
two places, the **projections** at the trunk's entry and exit — the
only rectangular weight matrices in the network:

```
parameters (n_params numbers)
     │
     ▼  entry projection     W is n_params × width: the one place
     │                       the vector grows to the working width
     │
     the trunk's residual blocks — every dense layer width → width
     │
     ▼  exit projection      W is width × n_dv: the one place the
     │                       vector becomes data-vector sized
data vector (n_dv numbers)
```

On the path the data travels, those two projections are the only
dimension changes anywhere in the model — the correction heads
rearrange and change basis, but never resize.

### `mlp`

The trunk. Every architecture is built on it, so this block is required.

An MLP — multilayer perceptron — is the simplest neural network: a
stack of dense layers with a nonlinear activation between them,
wrapped by the two projections defined above.

```
parameters (rescaled, section 2)
     │
     ▼  entry projection        n_params → width
     │
   ┌─┴───────────────────────────────────────────────┐
   │  residual block             repeated n_blocks   │
   │     │                                           │
   │     ├──────────────────────────┐                │
   │     ▼                          │                │
   │  dense → norm → activation     │  the identity  │
   │     │                          │  skip: the     │
   │     ▼                          │  block's input,│
   │  dense ────▶ + ◀───────────────┘  unchanged     │
   │     │                                           │
   │     ▼                                           │
   │  norm → activation                              │
   │     │                                           │
   │     ▼                                           │
   │  block output = input + correction              │
   └─┬───────────────────────────────────────────────┘
     │
     ▼  exit projection         width → data-vector length,
     │                          then one final learned scale
     │                          and shift
data vector (rescaled)
```

Each residual block adds its own input back to its output, so a block
only has to learn a correction. That is what keeps deep stacks easy to
train. The `norm` and `activation` slots between the dense layers are
exactly the next two subsections.

Two knobs. `width` is how many numbers each dense layer carries.
`n_blocks` is how many residual blocks are stacked.

```yaml
  mlp:
    width:    128
    n_blocks: 4
```

### `activation`

A dense layer can only take weighted sums, and a stack of weighted
sums is still a weighted sum — a straight-line map, however deep. The
activation is the step between dense layers that bends each number
individually; the bends are what let the network fit curved physics
at all.

The default family, `H`, is a bent line:

```
 H(x)
   │                            /
   │                          /      right of the bend: slope 1 —
   │                        /        large positive values pass
   │                      /          through unchanged
   │                    /
   │                  /
   │               _/
 ──┼─────___..--˙˙────────────────  x
 __│_..-˙
   │       left of the bend: slope gamma, a learned number;
   │       beta, also learned, sets how sharp the bend
   │       between the two tails is
```

**Learnable** means exactly that: gamma and beta are trained by the
same gradient descent that trains the weights, and every feature gets
its own independent pair. A **feature** is one of the `width` numbers
flowing through the layer — the `norm` subsection below draws it. The
network therefore does not just learn what to compute; it also learns
the shape of its own nonlinearity, feature by feature. All four
learnable families start from the same gentle line, `0.5·x`, and bend
away from it as training demands.

| `type` | learned numbers per feature | the shape |
|---|---|---|
| `H` | 2 | one bend — slope gamma to its left, slope 1 to its right |
| `power` | 3 | `H` plus a learned tail exponent: the tails may grow like $x^p$, with $p$ kept inside $[0.5, 1.5]$ |
| `multigate` | 3·`n_gates` + 1 | `n_gates` bends at learned positions — a slope schedule along x |
| `gated_power` | 3·`n_gates` + 2 | the multiple bends and the tail exponent together |
| `relu` | 0 | negatives cut to zero, positives pass — the textbook baseline |
| `tanh` | 0 | an S-curve, flat at both ends; those flat ends are the saturation trap the `norm` subsection explains, so pair it with `norm: per_feature` |

The block is `{type, n_gates}` or a bare type string; `n_gates` is
read only by the two multi-gate families. The exact formulas are in
the [activation appendix](#21-appendix-activation-functions).

```yaml
  activation:
    type:    H
    n_gates: 3
```

This sets one shared family for the whole model, trunk and head alike.
A `rescnn` / `restrf` head may pin its own family with
`model.cnn.activation` / `model.trf.activation`; leaving that key
absent means the head shares the trunk's. A pinned head needs a
frozen-trunk head phase, `head: activation:` is an alias for the same
pin, and the precedence and its warning are precedence
[A](#22-appendix-precedence--who-wins-when-settings-collide).

### `norm`

The normalization slot inside each trunk block. It sits between every
dense layer and its activation:

```
inside a residual block, every dense layer is followed by

  dense ──▶ norm ──▶ activation
             │
             g·x + b     g and b learned — a rescaling the
                         network can adjust as it trains
```

Why it is there: as training moves the weights, the numbers leaving a
dense layer can drift very large or very small, and an activation only
responds in a limited window. A value pushed far outside that window
lands where the activation's curve is flat — its output stops
changing, its gradient dies, and that part of the layer stops
learning. The failure is called **saturation**, and the norm is the
paper's guard against it:

```
                      the window where the
                      activation still bends
                           ┌─────────┐
values leaving       ●     │  ● ● ●  │      ●      the two outer
a dense layer              └─────────┘             values sit on the
                                                   flat tails: frozen
                                                   output, no gradient
                           ┌─────────┐
after g·x + b            ● │ ● ● ● ● │ ●           rescaled back to
                           └─────────┘             where the curve
                                                   bends — every value
                                                   can learn again
```

Three settings:

| `norm:` | learned numbers | what it can fix |
|---|---|---|
| `affine` — the default, absent = this | one pair $(g, b)$ shared by the whole layer: 2 | a global scale drift — the paper's choice |
| `per_feature` | one pair per feature: 2·width, 256 at width 128 | each feature's individual operating point — the escalation when one shared pair cannot hold every feature inside its window; pair it with `tanh` |
| `none` | 0 | nothing — an ablation |

```yaml
  norm: affine    # affine (default) | per_feature | none
```

A "feature" is one coordinate of the hidden vector flowing through the
trunk — the ResBlock width (`model.mlp.width`). Inside every ResBlock the
batch is a `(B, width)` tensor: B rows (cosmologies), width columns — the
columns are the features:

```
                 features (columns, width = model.mlp.width) ─▶
          ┌─  x_1   x_2   x_3   ...   x_128 ─┐
  B rows  │  (one row = one cosmology's       │
  (batch) │   hidden representation)          │
          └───────────────────────────────────┘

  per layer   :  g·x + b       one (g, b) pair for the whole tensor
  per feature :  g_i·x_i + b_i  one pair per column, shared by every row
```

Not a data-vector element, not a cosmological parameter, not a sample —
the same sense in which the activations' gamma and beta are
per-feature.

Only the trunk ResBlocks read this key — the transformer head
normalizes with its own internal LayerNorm, and the conv head has no
norm slot.

**Why no batchnorm.** Batch normalization is deliberately not offered:
its batch coupling would confound the batch-size and EMA experiments,
and its train/eval running-stats split — with buffers outside the EMA
weight average — risks baking a fixed mode under the compiled eval twin.
The paper prescribes the affine as batchnorm's replacement, and
`per_feature` is the escalation.

### `cnn` (name `rescnn`)

The conv head's premise: whatever the trunk gets systematically wrong
varies smoothly along the angular axis, so the fix should be built
from angular neighborhoods. The layer built for exactly that is the
**convolution** — a small set of learned weights slid along the data
vector and applied identically at every angular position:

```
one tomographic bin's stretch of the data vector, in theta order

   v1   v2   v3   v4   v5   v6   v7  ...
        └────────┬────────┘
          k1   k2   k3           the kernel: kernel_size learned
                 │               numbers — 3 drawn here, 11 in the
                 ▼               snippet below
       out4 = k1·v3 + k2·v4 + k3·v5

then the same three numbers slide one step right and produce out5,
then out6, and so on — one small weight set reused at every
position. A dense layer doing this job would hold a separate weight
for every pair of positions and would connect the largest angular
scales directly to the smallest.
```

Two properties follow. The correction is angular-local by
construction: each output reads only a `kernel_size`-wide window. And
the head stays tiny: the kernel is reused along the whole axis instead
of growing with the data-vector length.

#### Bins as channels

The data vector is not one long curve — it is many tomographic bins
laid end to end. The head rearranges it into a grid, one row per bin,
and the convolution runs on all rows at once:

```
theta-order data vector: the bins laid end to end

  [ bin 1 ●●●●●●●● │ bin 2 ●●●●●● │ bin 3 ●●●●●●● │ ... ]
        │
        │   a free rearrangement — numbers move, nothing multiplies
        ▼
              theta ─▶
   bin 1   ● ● ● ● ● ● ● ●
   bin 2   ● ● ● ● ● ● ○ ○      ○ = padding: shorter bins are
   bin 3   ● ● ● ● ● ● ● ○      zero-filled to the longest bin's
    ...                         length, and the pad slots are
                                dropped again on the way out
```

The rows are the conv's **channels** — the standard term for the
parallel signals one convolution mixes. At each angular position,
every output bin reads a `kernel_size`-wide window of every input
bin: local in angle, coupled across bins. That coupling is physical —
the bins share one angular grid, so covariance leaks between bins at
like angular scales.

#### Where the learned weights are — and where they are not

A textbook conv head bolted onto a vector output needs learned
adapters: a linear layer in to build the channels, an expansion to
some internal filter count, and a linear layer out to restore the
output size. Those adapters usually dominate the head's parameter
count. Here every adapter job is done by something free or frozen —
the graph is worth reading arrow by arrow:

```
trunk output y                    in the trunk's rescaled units
   │
   │  fixed basis change         a matrix precomputed once from the
   ▼                             data covariance — frozen, never
theta-order data vector          trained — into angular order
   │
   │  rearrange into the grid    free: numbers move, nothing
   ▼                             multiplies
(bins × theta) grid
   │
   │  n_blocks × [conv → activation]     the head's only learned
   ▼                                     weights; the last conv
corrected grid                           starts at exactly zero
   │
   │  rearrange back;            the same frozen matrix,
   ▼  undo the basis change      inverted
correction
   │
   ▼
out = y + gate · correction      gate: one learned number, starting
                                 small — the head fades in instead
                                 of disturbing the trained trunk
```

Nothing on that path resizes. The channels are the physical bins and
never expand; the convolution pads its ends so the angular length
never changes; the basis changes are fixed matrices computed once
from the covariance. The head's learned weights are the conv kernels,
their activations, and the gate — there are no adapter layers to eat
parameters. And because the last conv starts at zero, the correction
is zero at epoch 1: the model begins exactly equal to its trunk.

#### The knobs

| knob | what |
|---|---|
| `kernel_size` | the sliding window's width, odd; tuned as if the head had one block |
| `rescale_kernel` | shrink the per-block kernel as depth grows, keeping the whole stack's total view fixed at `kernel_size` |
| `groups` | cut the cross-bin mixing: `2` = xi+ and xi− never mix, the graph below; `3` / `6` on the factored head |
| `separable` | factor each conv into a per-bin angular filter plus a channel mix — the same sum, roughly `kernel_size`/2 times fewer weights |
| `film` | re-inject the cosmology as a per-bin scale and shift computed from the parameters — the paragraph below |
| `n_blocks` | stacked conv + activation blocks |
| `gate_init` | the gate's starting value — small, never 0: a zero gate passes no gradient and the head would never learn |
| `activation` | the head's own family; absent = share the trunk's |

`groups` uses the one physical cut the channel order offers — the
channels are the bins in data-vector order, all xi+ pairs first, then
all xi− pairs:

```
channels:   xi+ pair 1 .. P │ xi- pair 1 .. P

groups: 1   no cut — every output bin reads every bin
groups: 2   cut at the │ — xi+ and xi- never mix, and the
            head's conv weights halve
```

`film` addresses a blind spot: the head only ever sees the trunk's
output, so without it the head is one fixed correction map, identical
at every point of parameter space. With `film: true` each block also
receives the cosmological parameters, turned into one scale and one
shift per bin — the cosmology chooses which bins' corrections to
amplify and by how much. FiLM starts as an exact identity, so the
epoch-1 guarantee above is untouched.

```yaml
  cnn:
    kernel_size:    11
    rescale_kernel: false
    groups:         1
    separable:      false
    film:           false
    n_blocks:       1
    gate_init:      0.1
    # activation:        # optional: the head's own family
    #   type: gated_power
```

### `trf` (name `restrf`)

The conv head corrects within a window of neighboring angular scales.
The structure it cannot see is cross-bin at arbitrary separation: the
trunk's residuals correlate between different source-pair bins, and
no angular window covers that. The layer built for "everything may
look at everything, with learned, data-dependent weights" is
**attention**, the transformer's core.

The head first rearranges the data vector into the same bins × theta
grid the conv head uses. Each row — one tomographic bin's 26 angular
values, in this run — becomes one **token**, the unit attention works
with.

#### What attention computes

At the center is a bins × bins table of weights, rebuilt for every
sample from the tokens themselves:

```
        reads from:   bin 1   bin 2   bin 3   ...
   bin 1              0.81    0.14    0.02        each row says how
   bin 2              0.09    0.77    0.06        much that bin pulls
   bin 3              0.03    0.05    0.90        from every other
   ...                                            bin; rows sum to 1
```

Where the table comes from: three learned square matrices — `wq`,
`wk`, `wv` — turn each token into a **query**, what this bin is
looking for; a **key**, what this bin offers; and a **value**, what
it hands over if selected. Bin g's row of the table is large where
its query matches another bin's key, and what bin g receives is the
weighted mix of the values. None of the table is fixed: different
cosmologies produce different tokens and therefore different tables.

After attention has shared information across bins, each bin digests
what it received through its own private stack of dense layers —
`n_mlp_blocks` deep, every layer at the token width. Private weights
per bin is a deliberate deviation from the textbook transformer,
which shares one stack across all tokens. The bins are physically
distinct, and their unique weights are also what lets the model tell
them apart — the job a positional encoding does elsewhere, so this
head needs none. `shared_mlp: true` restores the textbook sharing as
an ablation.

#### One block, drawn

```
tokens (bins × 26)
   │
   ├──▶ LayerNorm ──▶ attention across bins ──▶ wo, zero-init ──┐
   │                                                            │
   +  ◀─────────────────────────────────────────────────────────┘
   │
   ├──▶ LayerNorm ──▶ each bin's private dense stack ───────────┐
   │                     (its last layer zero-init)             │
   +  ◀─────────────────────────────────────────────────────────┘
   │
   ▼
tokens out       = tokens in, exactly, at initialization
```

LayerNorm is the transformer's own normalization — a per-token
rescaling that keeps the attention table from saturating; it fills
the role `model.norm` fills in the trunk and is not configurable
here. Both branches end in a zero-initialized layer, so every block
is exactly the identity at initialization: the head's correction is
whatever the blocks add, the model equals its trunk at epoch 1, and
the same small learned `gate` fades the correction in — the conv
head's start, reproduced.

#### No adapters here either

A textbook transformer first embeds its input into learned token
vectors and projects back to the output shape at the end. In the
published CMB-emulator design those two adapters were the
parameter-heaviest layers of the whole network. This head has
neither: the tokens are the physical bin segments at their natural
width of 26 angular points, and the blocks' output is already in
data-vector layout — the same free-or-frozen principle as the conv
head's graph, applied to a transformer.

#### The knobs

| knob | what |
|---|---|
| `n_heads` | attention runs `n_heads` times in parallel on slices of the token — the graph below; must divide the token width, so 26 allows 1, 2, or 13 |
| `n_blocks` | stacked transformer blocks |
| `n_mlp_blocks` | depth of each bin's private dense stack; every layer at the token width, so there is no width knob — depth only |
| `n_tokens` | family runs only (CMB / BAOSN): re-segment the single spectrum into this many contiguous windows so attention has tokens to attend across (the token width becomes `ceil(n / n_tokens)`, which `n_heads` must divide). Rejected where physical bins already exist — cosmic shear's tomographic bins and the matter-power z slices ARE the tokens |
| `shared_mlp` | one shared stack for all bins — the textbook block, kept as an ablation |
| `film` | cosmology-aware per-bin scale and shift, exactly as `cnn.film` |
| `gate_init` | the correction gate's starting value — small, never 0 |
| `activation` | the head's own family; absent = share the trunk's |

```
one token = one bin, width 26            n_heads: 2 -> d_head = 13

    ┌──────────────┬───────────────┐
    │ head 1: 1-13 │ head 2: 14-26 │     the feature axis is sliced
    └──────┬───────┴───────┬───────┘     into n_heads equal parts:
           │               │             26 = n_heads x d_head, so
    G x G attention  G x G attention     n_heads must divide 26
    over all bins,   over all bins,      (1 | 2 | 13); G = the
    using slice 1    using slice 2       number of bins
           │               │
           └─── concat ────┴──▶ width 26 again; wo remixes
                                the heads into one token
```

The four attention matrices — `wq` / `wk` / `wv` and the output mix
`wo` — are 26 × 26 at any `n_heads`: same parameters, same cost. More
heads just means more, narrower attention tables per bin pair.

`compile_mode` (optional, flat) sets the CUDA `torch.compile` mode; the
defaults are precedence
[F](#22-appendix-precedence--who-wins-when-settings-collide).

```yaml
  model:
    name: restrf
    ia:   nla
    mlp:
      width:    128
      n_blocks: 4
    trf:
      n_heads:      2
      n_blocks:     1
      n_mlp_blocks: 2
      gate_init:    0.1
```

---

## 11. Two-phase schedule + the `trunk:` / `head:` blocks

A factored head (`rescnn` / `restrf`) can train in two phases: the trunk
alone, then the head:

```
phase "trunk"  (epochs 1 .. trunk_epochs)   head bypassed, trunk trains alone
      │  restore the best trunk weights
      ▼
set_train_phase("head" if freeze_trunk else "joint")
      │  freeze the trunk (default), or keep it training (joint fine-tune)
      ▼
phase "head" / "joint"  (the remaining nepochs - trunk_epochs)
      head from its zero-init identity, so the handoff is loss-continuous
```

The symmetric `trunk:` and `head:` blocks are **diffs** against the top
level: each one configures its own training pass, and any key left out of
the block keeps the run's top-level value. Eight keys may appear in either
block, and they override in two different ways:

| Keys in a phase block | How the override works |
|---|---|
| `lr` | An overlay: set `lr_base` or `warmup_epochs` for that pass and the other keeps its top-level value. `bs_base` is run-global and never appears here. |
| `scheduler` | A full replacement of the scheduler settings for that pass. The scheduler class itself never changes. |
| `loss`, `trim`, `focus`, `clip`, `rewind`, `ema` | A full replacement: state the whole block you want for that pass, including sub-keys. Nothing merges. |

The fine print is precedence
[B](#22-appendix-precedence--who-wins-when-settings-collide). One
asymmetry: the `head:` block may also carry `activation:`, an alias that
pins the head's own activation family. The same key inside `trunk:` is an
error; that rule is precedence
[A](#22-appendix-precedence--who-wins-when-settings-collide).

Single-phase models never break on a two-phase YAML. On any `resmlp`,
`train()` demotes the phase keys: `trunk:` merges into the top level, and
`head:`, `trunk_epochs`, and `freeze_trunk` are dropped. The same YAML
therefore drives both model families — what is in the trunk block is just
the global.

```yaml
  trunk_epochs:  1500
  freeze_trunk:  false
  head:
    lr:
      lr_base:       0.001
      warmup_epochs: 5
    scheduler:
      mode:     min
      patience: 10
      factor:   0.8
    loss:
      mode: berhu_capped
      berhu:
        knot: 0.2
        cap:  10
```

---

## 12. `pce`

Neural PCE (NPCE): before any network trains, fit a closed-form
polynomial approximation of the training set — the **base**
$B(\theta)$ — then train the `model.name` architecture as a
**refiner** $f(\theta)$ that corrects what the base misses. The base
is analytic: no network, no gradient descent, one least-squares pass
at staging. It captures the smooth, low-order dependence on the
cosmology; the refiner handles the rest — "trunk = PCE, head = any
SGD model". The fit runs on the training set in its rescaled form
from section 2, the same units the chi2 loss uses. The `pce:` block
is a top-level sibling of `data` / `train_args`; present = NPCE on,
absent = a plain run.

### What the base is made of

PCE stands for **polynomial chaos expansion**: a sum of fixed, known
polynomials of the input parameters. The family used here is the
**Legendre polynomials** — one curve per degree, nothing about them
learned or fitted. The fit only chooses which curves enter, and with
what coefficients:

```
degree 0        degree 1        degree 2        degree 3
the constant    the tilt        one bow         two bends

─────────           ╱           ╲     ╱          ╱╲
                  ╱              ╲   ╱          ╱  ╲    ╱
                ╱                 ╲_╱               ╲__╱

                       ... each degree adds one more wiggle
```

Before any polynomial is evaluated, each parameter is rescaled onto
the interval $[-1, 1]$ over its training range — the Legendre
polynomials' home interval, where they are **orthogonal**: each curve
carries information the others do not, which keeps the least-squares
fit stable as terms are added. At evaluation time, a point just
outside the training box is clamped to the interval's edge.

### One term = a product of one-parameter curves

With 12 parameters, one basis term multiplies together one Legendre
curve per parameter — most of them the trivial degree-0 constant:

```
term:  P2(Omega_m) · P1(sigma_8)        a bow in Omega_m times
                                        a tilt in sigma_8
written as one degree per parameter:

  alpha = ( 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
            │  │  └────────────┬────────────┘
            │  │               └── the other ten parameters absent
            │  └───── degree 1 in sigma_8
            └──────── degree 2 in Omega_m

total degree       2 + 1 = 3         capped by p_max
interaction order  2 parameters      capped by r_max
```

### Which terms are allowed

Three rules prune the candidate list before anything is fit. `p_max`
caps the total degree; `r_max` caps how many parameters may appear
together in one term. The third, the hyperbolic exponent `q`, decides
how degree may be spread across parameters — with `q` below 1,
spreading the same total degree over several parameters scores as
more expensive than concentrating it in one:

| term, total degree 4 | score at `q: 1` | score at `q: 0.5` |
|---|---|---|
| $x_1^4$ | 4 — kept | 4 — kept |
| $x_1^2 x_2^2$ | 4 — kept | 8 — dropped |
| $x_1 x_2 x_3 x_4$ | 4 — kept | 16 — dropped |

That preference is the sparsity-of-effects prior: smooth physics is
usually dominated by single-parameter trends plus mild low-order
interactions, so the many-parameter cross terms are the right ones to
drop first. Smaller `q` = sparser basis.

### What gets fit, mode by mode

The base does not fit the data vector entry by entry. The training
cloud is first split into its main directions of variation — an SVD,
the same idea as principal components: mode 0 is close to an overall
amplitude, and the modes after it are shape changes of decreasing
size. Each mode's amplitude is one number per training sample, and
those amplitudes are what the polynomials fit:

```
training set: parameters + data vectors, rescaled units
     │
     │  split the data-vector cloud       mode 0 ≈ the overall
     ▼  into its modes                    amplitude; mode 1, 2, ...
per-mode amplitudes z_0, z_1, ...         ever-smaller shape changes
     │
     │  per mode: least squares over the allowed terms — terms
     │  added greedily, each addition judged by its
     ▼  leave-one-out error
keep a mode in the base only if that error < loo_max
     │                                    │
     ▼                                    ▼
B(theta) = mean + the kept modes,    the rejected modes are left
rebuilt from their polynomial fits   for the refiner to learn
```

The **leave-one-out error** is the fit's error at each training
point, computed as if that point had been excluded from the fit — a
generalization test that needs no refit and no held-out data. It is
the gate everywhere above, because a mode kept with a poor fit
injects more error than it removes; every rejected mode is simply
left to the refiner, which corrects the full data vector and
backstops everything the gate drops.

The other hard-won rule is to keep the degree low. A high-degree
polynomial can pass near every training point and still oscillate
between them — Runge oscillation:

```
low degree:    ───●────●────●────●───     follows the smooth trend

                  ╱╲   ╱╲   ╱╲
high degree:   ──●──╲─●──╲─●──╲──●───     hits every point, wiggles
                     ╲╱   ╲╱   ╲╱         in between — wiggles the
                                          refiner must then spend
                                          capacity learning back out
```

`p_max` is therefore the smoothness knob, and the leave-one-out gate
— not the term cap — decides each mode's size.

### The combine form and the block

Two combine forms (`form`, required):

$$\mathrm{pred} = B(\theta) + f(\theta) \qquad
\mathrm{pred} = B(\theta) (1 + f(\theta))$$

with $B$ = the closed-form base and $f$ = the refiner. `residual`
adds the correction in the same rescaled basis the fit runs in and is
the usual choice. `ratio` multiplies in physical units and suits a
smooth low-order base — a multiplicative refiner has little leverage
where the base is near zero.

```yaml
pce:
  form: residual         # residual = B + f | ratio = B * (1 + f)
  # fit knobs (defaults = the values shown; all optional):
  # p_max:     4         # max total degree (the smoothness knob)
  # r_max:     2         # max interaction order (vars per term)
  # q:         0.5       # hyperbolic sparsity exponent in (0, 1]
  # k_max:     40        # max leading SVD modes to try
  # loo_max:   0.05      # keep a mode only if its relative LOO < this
  # max_terms: 30        # per-mode active-set cap
  # max_fail:  4         # stop after this many consecutive misses
```

The refiner is any `model.name` (`resmlp` / `rescnn` / `restrf`) with every
knob — its own two-phase schedule, per-head activation pins, `model.norm`, the
loss ladder (berhu included), trim / focus / clip / rewind / ema. `pce:` is
exclusive with `--rescale` and `model.ia` (each replaces the chi2 loss), and it
is structurally unsweepable — a top-level block, not a `train_args` leaf, so
one base per study; the collision rules are the
[precedence appendix](#22-appendix-precedence--who-wins-when-settings-collide).

---

## 13. Starting from a saved emulator: fine-tuning + transfer

Training normally starts from random weights. Two features reuse a trained,
saved emulator instead — and in both, epoch 0 is EXACTLY the saved emulator's
own prediction (a pre-train check refuses to train otherwise), so the run
improves on a proven starting point rather than a hopeful one.

**Which one to use:** fine-tuning adapts every weight of the old network, so
it fits small moves — the same physics on more data, or a couple of extra
parameters (LCDM -> w0waCDM). Transfer learning keeps the old network
completely frozen and trains a small parallel correction net, so it fits big
moves where the old capacity cannot stretch (many new parameters carrying new
physics, e.g. Early Dark Energy + w(z)-PCA amplitudes) — and it needs fewer
new training cosmologies, because everything the old emulator knows is kept
intact.

### Fine-tuning (`train_args.finetune`)

Point `from` at a saved emulator's path root (the `.h5` + `.emul` pair) and
drop the `model:` block — the architecture is read from the file, never
restated. New parameters in the training set (extra covmat columns, e.g.
`w0`, `wa`) are handled automatically: the old weights are kept exactly, and
the new inputs start with zero influence. Lower the learning rate through the
ordinary `lr:` block (about 10x below the original run), and keep a few
warmup epochs.

```yaml
train_args:
  finetune:
    from: projects/lsst_y1/chains/my_lcdm_run   # <root>.h5 + <root>.emul
    anchor: 1.0e-2      # optional: pull weights back toward the saved
                        # emulator (0.0 or absent = no pull). Columns for
                        # NEW parameters are never pulled. With an anchor,
                        # set optimizer weight_decay to 0.0.
  lr:
    lr_base:       5.0e-4
    bs_base:       64.0
    warmup_epochs: 5
```

Full example: `example_yamls/cosmic_shear_finetune_emulator.yaml`. Design
record: `notes/artifacts-inference-warmstart.md`.

### Transfer learning (`transfer:`)

The saved emulator (the "base") stays frozen; the `model:` block now
describes a NEW small correction net that sees the full new parameter space.
The two are combined element-wise, per data-vector template for the factored
NLA/TATT designs, before the amplitude combine. `form` picks the rule —
`gain` = base * (1 + correction), `sum` = base + correction — and `space`
picks where it acts (`physical` data-vector bins or the `whitened` training
coordinates; each form has a recommended space filled in automatically, and
choosing the other prints a note explaining the trade-off). The frozen base
is evaluated once per training row and cached, so training costs only the
small net; the saved result embeds the base, so the artifact reloads and
samples with no other file.

Transfer learning exists for the cosmolike and CMB data-vector families
only, and that restriction is permanent — a `transfer:` block on a
scalar, background, or matter-power config is a loud error; those
families [fine-tune](#fine-tuning-train_argsfinetune) instead.

```yaml
transfer:
  from: projects/lsst_y1/chains/my_lcdm_run
  form: gain            # gain = base*(1+r) | sum = base + r
  space: physical       # optional; defaults to the form's recommendation

train_args:
  model:                # the SMALL correction net (the base is untouched)
    name: resmlp
    mlp:
      width:    32
      n_blocks: 1
```

Full example: `example_yamls/cosmic_shear_transfer_emulator.yaml`. Design
record: `notes/artifacts-inference-warmstart.md`.

### Joint refinement (`transfer.refine`, optional stage 2)

After the correction converges, an optional second stage unfreezes the base
and trains both together: the base moves at a fraction of the run's learning
rate (`base_lr_scale`) and is pulled back toward its saved weights by the
`anchor` (both keys are required — an explicit `anchor: 0.0` states free
fine-tuning deliberately). The saved artifact keeps the ORIGINAL base, the
drifted base, and the drift itself, so you can always see how far the trusted
emulator moved to buy the extra accuracy.

```yaml
transfer:
  from: projects/lsst_y1/chains/my_lcdm_run
  form: gain
  refine:
    epochs:        200
    base_lr_scale: 0.01
    anchor:        1.0e-2
```

---

## 14. Scalar (derived-parameter) emulators

The emulator above maps cosmological parameters to a cosmic-shear DATA
VECTOR. A scalar emulator instead maps them to a small set of NAMED derived
parameters — H0, omegam, rdrag — one number each. The classic use lets a
sampler walk a fast variable while the slow map runs as an emulator: sample
the acoustic scale thetastar and emulate (omegabh2, omegach2, thetastar) ->
(H0, omegam), so cosmolike, which needs H0, keeps running. A separate driver,
`scalar_train_emulator.py`, trains one; there is no data vector, no mask, and
no cosmolike anywhere on this path.

```
sampled parameters (rescaled, section 2)
     │
     ▼  resmlp trunk            the same architecture as section 10,
     │                          just narrower — a scalar map is easy
standardized outputs
     │
     ▼  undo the standardization
H0, omegam, ...                 one physical number each, served by name
```

**Inputs and outputs are both columns of one parameter file.** The inputs
are the covmat-header names, rescaled into the network's training units
exactly as on a data-vector run (section 2). The outputs are the columns you
name in `data.outputs`, each standardized — shifted to zero mean and scaled
to unit variance. The outputs live on wildly different scales, H0 near 70
and omegam near 0.3, and standardizing puts each on the same footing before
the network sees it. The `.txt` needs its getdist `.paramnames` sidecar
beside it, since the outputs are usually derived columns located by name.

```yaml
data:
  train_params: chain_thetastar_lcdm.1.txt    # needs a .paramnames sidecar
  train_covmat: chain_thetastar_lcdm.covmat    # header = the input names
  val_params:   chain_thetastar_lcdm_val.1.txt
  outputs:                                     # the derived columns to emulate
    - H0
    - omegam
  n_train:    100000
  n_val:      20000
  split_seed: 0
```

The model is a plain trunk (`name: resmlp`); the conv and transformer heads
correct along an output coordinate axis (theta / ell / z / k — every other
family has one), and a scalar output is a set of named values with no axis
between them, so `rescnn` / `restrf` are a loud error here. Everything
else — the loss ladder, trimming,
focus, EMA, the L2-SP anchor — works unchanged, since they act on a
per-sample error. A scalar map is cheap, so small widths and a few hundred
epochs are plenty. The physical-window `param_cuts` are optional on this
path, because a parameter chain is already the target distribution.

**In an MCMC, the theory block reads what it provides from the file.** One
generic class, `emul_scalars`, serves any scalar emulator: it lists each
saved emulator's path root and nothing else — the required inputs and the
provided outputs both come from the `.h5`, never a hand-typed list. Point it
at several roots and it provides their union. Three misconfigurations are
loud errors at startup:

| the error | why |
|---|---|
| two emulators provide the same output name | each derived parameter must come from exactly one emulator |
| one emulator's output is another emulator's input | chaining scalar emulators is out of scope |
| a data-vector emulator in the list | `emul_scalars` serves scalar artifacts only — a data-vector emulator belongs in `emul_cosmic_shear`'s list |

```yaml
theory:
  emul_scalars:
    python_path: ./external_modules/code/emulators_code_v2/cobaya_theory/
    extra_args:
      device: cuda
      emulators:
        - projects/lsst_y1/emulators/thetaH0/emul_v2
        - projects/lsst_y1/emulators/rdrag/emul_v2
```

Full example: `example_yamls/scalar_emulator.yaml`. Design record:
`notes/families-scalar-cmb.md`.

---

## 15. Emulating CMB spectra (TT / TE / EE / phi-phi)

A CMB emulator maps cosmological parameters to one spectrum's C_ell
values — the angular power spectrum of the cosmic microwave background
(TT = temperature, TE = temperature-polarization cross, EE = E-mode
polarization, phi-phi = the lensing potential), on a fixed multipole
grid l = 2..lmax. One emulator learns ONE spectrum; a full set is four
artifacts. Everything rides the same training stack as the data-vector
emulators — the losses, trimming, focal weighting, EMA, clip / rewind,
fine-tuning all compose unchanged — because the loss exposes the same
per-sample chi2 interface. The correction heads apply too ([section
10](#10-model)): the CMB whitening keeps the multipole order, so
`rescnn` slides its kernel along ell directly, and `restrf` re-segments
the spectrum into `model.trf.n_tokens` contiguous windows and attends
across them — the tokenization of the attention-based CMB emulators
(arXiv 2505.22574, which finds attention cuts the outlier count vs a
plain MLP at this exact task).

The pipeline, end to end:

    dataset_generator_cmb.py            compute_cmb_covariance.py
    (CAMB through cobaya, one call      (Motloch & Hu 1709.03599 eqs 1-7,
     per sampled cosmology)              one fiducial-LCDM CAMB call)
          |                                   |
          |  dvs_*_tt.npy (+ te/ee/pp)        |  cmbcov_lcdm.npz
          |  params_*.1.txt + sidecars        |  (sigma_ell per spectrum,
          v                                   v   fiducial C_ell, provenance)
    +---------------------------------------------------+
    |  cmb_train_emulator.py                              |
    |  (a data.cmb block): whiten each multipole by       |
    |  its error bar sigma_ell, impose the amplitude      |
    |  law, train the model (a ResMLP trunk, or the       |
    |  rescnn / restrf correction heads, section 10)      |
    +---------------------------------------------------+
          |
          |  emulator_tt_*.h5 + .emul   (the artifact)
          v
    cobaya_theory/emul_cmb.py    serves get_Cl to any cobaya likelihood

### The covariance file (why a separate script)

For cosmic shear the loss covariance comes from cosmolike. A CMB
spectrum's covariance is analytic instead: the Gaussian variance of one
measured C_ell is (Motloch & Hu 1709.03599, eq 3)

    var(C_ell) = 2 / [(2l+1) fsky] * (C_ell + N_ell)^2

where N_ell is the instrumental noise spectrum built from the detector
noise level (in muK-arcmin) and the beam width (eq 1). The script
`compute_data_vectors/compute_cmb_covariance.py` computes this once, on
a fiducial LCDM cosmology at high CAMB accuracy, and writes one .npz the
training consumes; the optional non-Gaussian lensing terms (eq 6) sit
behind a flag, off by default. What you state in its YAML is the
experiment: the noise level, the beam, the sky fraction.

### The imposed amplitude law

The primary CMB spectra scale almost exactly as A_s e^(-2 tau) (the
primordial amplitude damped by reionization). Rather than making the
network learn that known scaling, the training can impose it: with

```yaml
  cmb:
    spectrum: tt
    covariance: cmbcov_lcdm.npz
    amplitude_law: as_exp2tau
    as_name:       As
    tau_name:      tau
```

the target the network sees is C_ell * exp(2 tau) / A_s — the SHAPE
only — and the emulator multiplies the law back on the way out. A_s and
tau are read from named parameter columns of the training dump (As must
be the linear amplitude, which the generator samples directly). Set
`amplitude_law: none` to learn the raw C_ell instead (and drop the two
names). The law is stored in the artifact by name, so a saved emulator
always knows its own convention.

### The roughness penalty (optional)

CMB spectra are smooth in l; short-period wiggles in the emulator
residual are network artifacts, never physics. The optional loss term

```yaml
  loss:
    mode: sqrt
    roughness:
      lam:        0.1   # weight; absent block = the term does not exist
      period_cut: 50    # penalize residual oscillation periods below this
```

adds, per training sample, `lam` times the short-period content of the
residual (a high-pass filter at `period_cut` multipoles) to the chi2
before the usual reduction. It acts on the residual — prediction minus
truth — so a perfect prediction pays nothing, however sharp or smooth
the true peaks: it cannot bias the lensing-induced peak smoothing,
whose period (~200-300, the acoustic spacing) the filter passes
untouched.

### Serving the spectra in an MCMC

```yaml
theory:
  emul_cmb:
    python_path: ./cobaya_theory/
    extra_args:
      device: cuda
      emulators:
        - chains/emulator_tt_resmlp_ntrain50000
        - chains/emulator_ee_resmlp_ntrain50000
```

Each path root declares its own spectrum, multipole range, and units
(they are stored in the artifact — nothing is restated in the YAML).
A likelihood that requests a spectrum no artifact provides, or
multipoles beyond an artifact's training grid, fails loudly at startup.
get_Cl serves raw C_ell in the dump units (muK^2; phi-phi
dimensionless), zero below l = 2.

### Fine-tuning

A CMB emulator warm-starts from a saved CMB emulator of the same
spectrum, law, and covariance file, exactly like the data-vector
fine-tune: add

```yaml
train_args:
  finetune:
    from: chains/emulator_tt_resmlp_ntrain50000
```

and delete the `model:` block (the architecture is inherited). Epoch 0
reproduces the source exactly; training then refines it on the new
dump.

### The diagnostics pages

`--diagnostic` on a data.cmb run appends two CMB pages to the usual
PDF: per-multipole residual bands (fractional AND in error-bar units —
read the error-bar panel for TE, which crosses zero) with the
worst-cosmology overlay, and the residual's short-period wiggle content
(what the roughness term sees). The example config is
`example_yamls/cmb_emulator.yaml`; the gates are cmb-identity and
cmb-smoke on the board.

---

## 16. Emulating the expansion history (H(z), BAO and SN distances)

Only H(z) is a network; every distance is known physics computed from
it. The BAOSN family serves the background — the Hubble rate and the
comoving / angular-diameter / luminosity distances — to BAO and
supernova likelihoods from TWO small artifacts:

    the "Hubble" artifact                the "D_M" artifact
    H(z) on the SN range, z in [0, 3]    the comoving distance, trained
       │                                 directly on the recombination
       │  emulator/background.py:        window z in [1000, 1200] (the
       │  chi(z) = integral of c/H       CMB-distance anchor)
       ▼  (cumulative Simpson)
    D_C = chi, D_A = chi/(1+z),
    D_L = chi*(1+z)   (flat)

Nothing is emulated between the two windows — no likelihood queries
that desert — and a query there is a loud error, never a silent bridge.
The training target for H(z) is log(H + offset) (the `log_offset` law,
persisted in the artifact); D_M trains raw (`none`). Dumps come from
`compute_data_vectors/dataset_generator_background.py`: one
background-only CAMB evaluation per sampled cosmology yields BOTH
quantities and the grids ride beside the dumps as `_z.npy` sidecars.
The full training surface applies here unchanged — the loss ladder,
trimming, focal weighting, EMA, clip / rewind, fine-tuning, and the
correction heads ([section 10](#10-model)): the standardization keeps
the z order, so `rescnn` slides its kernel along z and `restrf`
re-segments the grid into `model.trf.n_tokens` attention windows.

```yaml
data:
  grid:
    quantity: Hubble        # or D_M for the recombination artifact
    units:    km/s/Mpc      # Mpc for D_M
    law:      log_offset    # none for D_M
    offset:   0.0
    z_file:   dvs_train_background_unifs_h_z.npy
```

Training runs through `baosn_train_emulator.py`, a thin wrapper over
the shared train driver that pins the family; the full commented
config is `example_yamls/baosn_hubble_emulator.yaml`.

Serving in an MCMC pairs the two artifacts in one theory block (rdrag
comes separately, from the
[scalar-emulator family](#14-scalar-derived-parameter-emulators)):

```yaml
theory:
  emul_baosn:
    python_path: ./cobaya_theory/
    extra_args:
      device: cuda
      emulators:
        - chains/emulator_hubble_resmlp_ntrain50000
        - chains/emulator_dm_resmlp_ntrain50000
```

get_Hubble (km/s/Mpc or 1/Mpc), get_comoving_radial_distance,
get_angular_diameter_distance (+ the two-redshift variant), and
get_luminosity_distance are served piecewise by query redshift. V1 is
flat-only (a sampled omk is a loud error; the legacy curvature formula
was dimensionally wrong and is not reproduced). Fine-tuning works per
artifact (same quantity, grid, units, and law; the model: block is
inherited). `--diagnostic` adds the per-redshift residual bands and,
for the Hubble artifact, the derived-distance page computed through the
real integration pipeline. Gates: bsn-identity / bsn-smoke — the smoke
checks the served values against CAMB's own background, so it is the
strongest end-to-end test in the board.

---

## 17. Emulating the matter power spectrum (hybrid inference, EMUL2)

The MPS emulators CORRECT an approximate formula. The syren
(symbolic_pofk) expressions — vendored in-repo under `syren/`, so
nothing needs installing and the formula can never drift under a
package upgrade — give an analytic P(k, z); the network learns only
the residual:

    target = log( P(k, z) / P_syren(k, z; params) )

so the amplitude and shape it must capture are gentle, and the exact
formula is multiplied back at inference. Two artifacts serve
everything:

    the "pklin" artifact               the "boost" artifact
    corrects the syren linear          corrects syren-halofit's
    formula -> P_lin(k, z) [Mpc^3]     boost -> B = P_nl / P_lin
                        \                /
                         P_nl = B * P_lin

Dumps come from `compute_data_vectors/dataset_generator_mps.py`: one
CAMB call per sampled cosmology writes the raw surfaces AND the syren
base beside them (the training divides the base out once, at staging,
and never recomputes it — together with the vendored `syren/` the
base is pinned twice). The (z, k)
grids ride as `_z.npy` / `_k.npy` sidecars and persist into the
artifact; V1 trains on a thinned k grid (`k_stride`, top edge always
kept) and the served interpolator fills between kept points. The full
training surface applies here unchanged, correction heads included
([section 10](#10-model)): the flattening is z-outer, so `rescnn` gets
the z slices as conv channels and slides along k (mixing redshifts at
like k), and `restrf` gets one attention token per z slice. One
physical fact the training handles for you: below the nonlinear scale
the boost is 1 for every cosmology, so those grid points carry no
signal under any law — the geometry pins them (the served value is
exactly the training constant: the analytic base under a syren law,
the constant itself under law `none`) instead of failing on an
unlearnable column, and reports how many it pinned at startup.

```yaml
data:
  grid2d:
    quantity: boost           # or pklin for the linear artifact
    units:    dimensionless   # Mpc3 for pklin
    law:      syren_halofit   # syren_linear for pklin; none = raw
    train_base: dvs_train_mps_unifs_boost_base.npy
    val_base:   dvs_val_mps_unifs_boost_base.npy
    z_file:     dvs_train_mps_unifs_z.npy
    k_file:     dvs_train_mps_unifs_k.npy
    k_stride:   10
```

Training runs through `mps_train_emulator.py`, a thin wrapper over
the shared train driver that pins the family; the full commented
config is `example_yamls/mps_boost_emulator.yaml`.

**Hybrid inference (EMUL2):** cosmolike's `use_emulator: 2` mode
consumes EMULATED CAMB PRODUCTS instead of a full data-vector emulator:
P(k, z) from `emul_mps`, distances and H(z) from `emul_baosn`
([section 16](#16-emulating-the-expansion-history-hz-bao-and-sn-distances)),
r_drag from the
[scalar-emulator family](#14-scalar-derived-parameter-emulators) —
three theories in one sampling YAML. `emul_mps` serves `get_Pk_grid` /
`get_Pk_interpolator` (linear and nonlinear) through the
CAMB-compatible interpolator (adapted from CAMB by Antony Lewis), so a
likelihood written against CAMB's provider needs no change:

```yaml
theory:
  emul_mps:
    python_path: ./cobaya_theory/
    extra_args:
      device: cuda
      emulators:
        - chains/emulator_pklin_resmlp_ntrain50000
        - chains/emulator_boost_resmlp_ntrain50000
```

The acceptance experiment for the unit is the full EMUL2 evaluate run
with all three theories; the ready-to-fill config ships as
`cobaya_theory/EXAMPLE_EMUL2_EVALUATE.yaml` (the legacy
EXAMPLE_EMUL2_EVALUATE1.yaml pattern with the v2 adapters).
Fine-tuning works per artifact (same quantity, law, and grids).
Transfer learning is exclusive to the cosmolike and CMB data-vector
families and is permanently out here. Gates: mps-identity / mps-smoke.

---

## 18. Generating the training set

The `data` block (section 3) names the training dumps. This section is
where they come from: the `compute_data_vectors/` generators — MPI +
emcee + cobaya tools that draw cosmologies, compute each one's
training targets through the physics code, and write exactly the files
the trainer reads. One shared core (`generator_core.py`) owns the CLI,
the sampling, the MPI farm, and the checkpointing; each family adds a
thin driver that states its physics requirements and its file store:

| generator | family (README section) | truth code | writes |
|---|---|---|---|
| `dataset_generator_lensing.py` | cosmic shear (2–13) | cosmolike | one dv `.npy` + the params/covmat/ranges sidecars |
| `dataset_generator_cmb.py` | CMB spectra (15) | CAMB | four spectra files `_tt`/`_te`/`_ee`/`_pp.npy` |
| `dataset_generator_background.py` | expansion history (16) | CAMB (background) | the `_h`/`_dm` pair + `_z.npy` grid sidecars |
| `dataset_generator_mps.py` | matter power (17) | CAMB | the `pklin`/`boost` surfaces (+ their syren `_base` files) + `_z`/`_k` sidecars |

The CMB family needs one more file the trainer consumes: the analytic
per-multipole covariance, computed once on a fiducial cosmology by
`compute_data_vectors/compute_cmb_covariance.py` (section 15 walks its
physics). Everything below — the sampling modes, the tempered
posterior, the knobs, the machinery, the output contract — is the
shared core and applies to every generator; the lensing driver is the
worked example.

**The goal.** An emulator is only trustworthy inside the cloud of cosmologies
it was trained on, so the training density must cover *where chains will
explore* — which is broader than the posterior itself. A sampler that walks to
the edge of the posterior must still land inside the training support, so the
generator samples a deliberately widened distribution, not the posterior.

**Two sampling modes (peers, chosen by `--unif`).**

- Gaussian / tempered (`--unif 0`, the default): draws follow the tempered
  posterior below — a training cloud shaped like a widened posterior, dense
  where chains spend time.
- Uniform (`--unif 1`): draws are uniform inside the (temperature-stretched)
  hard bounds — a flat cloud filling the whole box, with no posterior shaping,
  and `lnp` is set to 1 (the rows carry no importance weight). Even uniform
  sampling still needs `--temp`: the temperature sets the hard-boundary stretch
  for parameters whose priors are Gaussian or unbounded (each such bound is
  widened by `temp * width / 5`), so without it those parameters have no box to
  be uniform in. Uniform outputs tag as `_<probe>_unifs` instead of
  `_<probe>_<T>`, so the trainer's `data:` filenames differ accordingly.

Which to use: tempered for production training sets (density where chains
explore); uniform for coverage studies or stress tests far from the posterior.

**The tempered posterior (`--unif 0`).** The generator samples

$$\log p_T(\theta) = \frac{1}{T}\left[ -\tfrac{1}{2}(\theta-\theta_0)^\top \tilde\Sigma^{-1} (\theta-\theta_0) + \log \pi(\theta)\right]$$

legend: theta = the sampled parameter vector; theta_0 = the fiducial
(`train_args.fiducial` in the generator YAML); T = the temperature (`--temp`,
also the `_cs_<T>` tag in every output filename — the same tag the trainer's
file names and the drivers' `t<T>` run tag parse); pi = the cobaya prior (hard
bounds respected, infinite-prior bounds stretched by `temp * width / 5`);
Sigma-tilde = the Fisher / params covmat (`params_covmat_file`) with its
correlations clipped to at most `maxcorr` (default 0.15). The whole bracket is
divided by T, so the effective covariance is `T * Sigma-tilde` — wider than the
posterior.

**Why each knob.**

| Knob | Why it exists |
|---|---|
| `--temp` | Flattens the likelihood by the temperature T, so the training cloud extends past the posterior. At T = 1 the cloud would hug the posterior, and a chain that reaches the posterior's edge would step outside the training support. Larger T widens the cloud. |
| `--maxcorr` | Fills the volume *perpendicular* to the degeneracy directions. A raw Fisher covmat is a thin pancake along the degeneracy, and the emulator needs volume there, not a line. Clipping the off-diagonal correlations fattens the pancake. |
| `--boundary` | Below 1, shrinks the validation and test sets *inside* the training support. Accuracy degrades at the cloud's edge, so the validation set must not sit on it. |

**The machinery.** emcee samples log p_T with differential-evolution moves
(`DEMove` 90%, `DESnookerMove` 10%); the chain is de-duplicated and reduced to
`--nparams` points, with the autocorrelation time reported. An MPI master hands
parameter rows to workers, each of which computes the data vector through the
cobaya model (CAMB + cosmolike, per the generator YAML). The run is
checkpointed (`--freqchk` / `--loadchk` / `--append`): a failed evaluation is
zeroed and flagged in the failfile, and recomputed by rerunning with
`--loadchk 1`.

**The output contract** (this is where every loop closes; `<T>` is the
temperature, or `unifs` for a uniform run):

| file | content | consumed by |
|---|---|---|
| `<paramfile>_<probe>_<T>.1.txt` | columns `weights`, `lnp`, `<params>`, `chi2*` | the trainer's staging slice (it drops the leading `weights` / `lnp` and the trailing `chi2*`) |
| `<paramfile>_<probe>_<T>.paramnames` | first column = the cobaya parameter names | ParamGeometry names, then the h5, then `get_requirements` (the naming loop) |
| `<paramfile>_<probe>_<T>.covmat` | the parameter covmat | the trainer's `data.train_covmat` — the input whitening basis |
| `<paramfile>_<probe>_<T>.ranges` | the sampled bounds | getdist plotting of the training cloud |
| `<datavsfile>_<probe>_<T>.npy` | the stacked training targets | the trainer memmaps it as `data.train_dv` (per-family store: the table above lists each family's file set) |
| `<failfile>_<probe>_<T>.txt` | the flagged failed rows | `--loadchk 1` reruns to recompute them |

**Two `train_args`, named loudly.** The generator YAML is a *cobaya* YAML with
its own `train_args` block (`probe`, `ord`, `fiducial`, `params_covmat_file`).
This is unrelated to the *emulator trainer's* `train_args` (section 2): two
stages, two schemas. The generator's `train_args` configures the cobaya model
that produces the dumps; the trainer's `train_args` configures the network that
learns them.

**Run it** (`$D` is the driver-folder shorthand from [section 1](#1-run-it);
the script's own header keeps a `roman_real` example verbatim):

```bash
mpirun -n 10 --report-bindings \
  python $D/compute_data_vectors/dataset_generator_lensing.py \
    --root projects/lsst_y1/ --fileroot emulators/nla_cosmic_shear/ \
    --nparams 10000 --yaml w0wa_takahashi_cs_cnn.yaml \
    --datavsfile w0wa_takahashi_dvs_train \
    --paramfile  w0wa_takahashi_params_train \
    --failfile   w0wa_takahashi_params_failed_train \
    --unif 0 --temp 64 --maxcorr 0.15 --freqchk 2000 --boundary 1.0
```

---

## 19. Appendix: the pipeline

The goal is to replace an expensive physics code with a network that maps a
handful of cosmological parameters to the cosmic-shear data vector, fast enough
to call inside a cosmological inference and accurate enough that the data
vector's [**chi2**](#20-appendix-the-chi2-metric-mahalanobis) — its distance from
truth measured in the data covariance (a Mahalanobis distance; see the appendix),
the quantity inference actually cares about — stays small. Two ideas run through the
whole pipeline. **Whitening**: both the inputs and the outputs are rotated and
rescaled so the network sees a decorrelated, unit-variance, well-conditioned
problem instead of raw correlated numbers. **The chi2 metric**: training and
evaluation are judged in the covariance's natural geometry, not in raw
per-element error, because that is what an analysis uses.

```
cosmological parameters
   │   geometries/parameter.py   center, rotate, unit-scale          (whiten in)
   ▼
whitened inputs
   │   designs/plain.py          ResMLP, ResCNN, or ResTRF
   ▼
whitened data vector
   │   geometries/output.py      un-whiten + scatter to full length  (whiten out)
   ▼
physical residual vs truth
   │   losses/core.py            contract with the inverse covariance
   ▼
chi2  =  r^T Cinv r
```

**1. Stage the data** (`data_staging.py`). The training set is a large dump of
`(parameters, data vector)` pairs the physics code wrote to disk. The dump is far
too big to hold in RAM, so it is memmapped and read in slices; physical cuts
drop the sparse corners no real posterior visits (the high-`omega_b h^2`
corner, and both tails of the `omegam^2 h^2` window when configured),
and only `N_train` rows are kept. The result is a "source" dict (`C`, `dv`,
`idx`) the rest of the pipeline consumes.

```
1.  Stage the data                            load_source · data_staging.py

      dv dump (.npy)                    params table (.txt)
          │                                  │
   np.load(mmap_mode=r)              loadtxt[:, 2:-1]
   never loaded whole                drop weight / lnp / chi2 cols
          │                                  │
          └────────────────┬─────────────────┘
                           ▼
                 seeded shuffle             randperm(n, gen)   (split_seed)
                           ▼
                 physical cuts              phys_cut_idx: keep lo < omega_b h^2 < omegabh2_hi
                           ▼                 + lo < omegam^2 h^2 < hi (lower bounds
                           │                 + window optional); omegab/omegam/H0
                           │                 columns found by name
                 keep N_train               idx = phys[:n_train]  (absolute, post-cut)
                           ▼
            stage_source:  subset bytes < ram_frac · available RAM ?
                 ┌─────────┴──────────┐
                yes                    no
                 ▼                     ▼
       ┌────────────────────┐  ┌────────────────────┐
       │ materialize the    │  │ keep the memmap    │
       │ compact subset,    │  │ + global idx       │
       │ reindex → arange   │  │ (stream from disk) │
       └────────────────────┘  └────────────────────┘
                 └─────────┬──────────┘
                           ▼
       source dict  { C, dv, idx  (+ C_mean, dv_mean — train only) }
                    ──────────────────────────────────────────────▶  build_loaders
```

The local `arange` reindex is the trick: every consumer reads `C` / `dv` only
through `idx`, so it does not matter whether `idx` points into the full memmap or
the compact in-RAM subset — the pipeline is identical either way.

**2. Whiten the inputs** (`geometries/parameter.py`). Raw cosmological parameters
are correlated and span wildly different scales. `ParamGeometry` centers them,
rotates into the parameter-covariance eigenbasis, and scales each direction to
unit variance, so the network receives decorrelated, unit-variance inputs rather
than strongly correlated physical numbers.

```
2.  Whiten the inputs                          ParamGeometry · geometries/parameter.py

      raw params  θ                            (B, n_param)  physical, correlated
          │
          │  − center                          c = training mean
          ▼
      centered
          │  @ evecs                           rotate into the param-covmat eigenbasis
          ▼                                     evecs, √λ = eigh(covmat)   (from_covmat)
      decorrelated
          │  / √λ                              scale every axis to unit variance
          ▼
      whitened input  X  ────────────────────▶ model sees decorrelated, σ = 1 inputs

      encode(θ) = (θ − c) @ evecs / √λ          decode = (X · √λ) @ evecsᵀ + c  (exact inverse)
```

**3. Whiten the output, and keep the metric** (`geometries/output.py`). The data
vector is *masked* (the analysis keeps only some entries) and strongly
correlated. `DataVectorGeometry` *squeezes* to the unmasked entries and whitens
them in the data-covariance eigenbasis, so every network output is decorrelated
and equally hard to fit. The same object holds `Cinv`, the masked inverse
covariance the chi2 contracts against — geometry and metric live together.

```
3.  Whiten the output, keep the metric         DataVectorGeometry · geometries/output.py

      raw data vector  d                        (B, total_size)  full 3x2pt
          │
          │  squeeze  d[:, dest_idx]            keep only the unmasked entries → (B, n_keep)
          ▼
          │  − center                           c = training mean of the kept entries
          ▼
          │  @ evecs / √λ                       whiten in the kept-block cov eigenbasis
          ▼                                      evecs, √λ = eigh(kept cov)
      whitened target  t  ◀──────────────────── network predicts  ŷ ≈ t
                                                 decorrelated, σ = 1 → every output equally hard

      ── the loss un-whitens, and scores the true metric ──────────────────────
          ŷ − t  ──unwhiten──▶  r               r = physical residual  (· √λ) @ evecsᵀ
                                    │
                χ² = rᵀ · Cinv · r  ◀┘            full masked precision (geom.Cinv_sq)
                                                 for full whitening this equals ‖ŷ − t‖²
                                                 (the whitening basis is the χ² basis)
```

**4. Build the loss** (`losses/core.py`). `make_chi2` wraps the output
geometry in a chi2 — the error metric of [appendix 20](#20-appendix-the-chi2-metric-mahalanobis),
which weighs each residual by how well the survey can measure it.

The network never sees raw data vectors. It is trained on *whitened*
targets: each data vector rescaled so that every component matters equally
to the chi2. That choice does two jobs at once. Ordinary squared error on
the whitened targets already equals the chi2, so the training objective is
the physics metric with no extra weighting. And because no component
dominates, the optimization stays numerically well behaved.

The loss also carries the optional robustness schedules: `trim` drops the
worst-fit points early in training, and `focus` up-weights the points that
stay hard. Both have their own sections ([7](#7-trim) and [8](#8-focus)).

```
4.  Build the loss                            make_chi2 · losses/core.py

      cosmolike cov / mask / inv-cov
                │  DataVectorGeometry.from_cosmolike   (one cosmolike read + one eigh)
                ▼
         geom   (built once)        owns squeeze · center · whiten · decode · Cinv
                │
                │  wrap  (composition: the loss HAS-A geom — self.geom, not inheritance)
                ▼
       rescale = ?
      ┌──────────────────┬─────────────────────────┬──────────────────────────┐
    "none"            "rescaled" (A)             "residual" (B)
      ▼                  ▼                          ▼
┌──────────────┐  ┌────────────────────┐    ┌────────────────────┐
│ CosmolikeChi2│  │ RescaledChi2       │    │ ResidualBaseChi2   │
│ plain masked │  │ R divides ŷ →      │    │ R moves the        │
│ Mahalanobis  │  │ diag(1/R) in grad  │    │ baseline; plain χ² │
└──────────────┘  └────────────────────┘    └────────────────────┘
 needs_params      needs_params = True          needs_params = True
  unset → False    + build_shear_angle_map(geom) + configure_rescaling(...)
                │
                ▼
        chi2fn  ─────────────────────────────────▶  run_emulator
        forwards encode / decode / dest_idx / total_size → self.geom
        pipeline branches on getattr(chi2fn, "needs_params", False)   (not isinstance)
```

The two ideas the diagram is built around: **composition** (one `geom` built once,
wrapped by whichever loss — never re-read, never inherited) and the
**`needs_params` capability flag** (a future param-aware loss just sets the flag;
nothing branches on `isinstance`).

**5. Choose the model** (`designs/plain.py`, `designs/ia.py`).

`ResMLP` is the baseline: an input projection, a stack of residual blocks,
an output projection. `ResCNN` adds a 1D convolution on top of the ResMLP
trunk, working along the angular axis where neighbouring data-vector
entries vary smoothly. `ResTRF` does the same job with a small transformer
whose tokens are the tomographic bins.

Two independent YAML keys pick the class. `train_args.model.name` chooses
the architecture. The separate `train_args.model.ia` key layers a factored
intrinsic-alignment design on top of it — the model then emits a few
templates and the loss combines them in closed form, so the IA amplitudes
never enter the network at all. Omit `ia` for the plain emulator.

| `model.name` | plain (`ia` omitted) | `ia: nla` (1 amplitude, 3 templates) | `ia: tatt` (3 amplitudes, 10 templates) |
|---|---|---|---|
| `resmlp` | `ResMLP` | `TemplateMLP` | `TemplateMLP` |
| `rescnn` | `ResCNN` | `TemplateResCNN` | `TemplateResCNN` |
| `restrf` | `ResTRF` | `TemplateResTRF` | `TemplateResTRF` |

In the factored variants the correction head works per template: the gated
convolution of `TemplateResCNN`, or the bin-token transformer of
`TemplateResTRF`, corrects each template before the closed-form combine.

**6. Feed the GPU** (`batching.py`). The staged data may or may not fit in GPU
memory, so the loaders pick a regime — hold the whole encoded set resident on the
GPU if it fits, otherwise stream it from RAM, or from the disk memmap, a chunk at
a time — and hand the training loop two closures (`load_C`, `load_dv`) that hide
which regime is in play, so the loop code is identical no matter the data size.

```
6.  Feed the GPU                              _build_loaders_one · batching.py

     one source's data vectors                budget = free VRAM (CUDA) | GPU_MEM (else)
               │
               │  encoded set fits?    enc_dvs + resident < 0.8 · budget
       ┌───────┴───────────────────────────────────┐
      yes                                           no
       │                                    dv still an in-RAM array?
       │                                ┌───────────┴───────────┐
       │                               yes                      no (np.memmap)
       ▼                                ▼                        ▼
 ┌───────────────┐             ┌──────────────────┐    ┌──────────────────┐
 │ Regime 1      │             │ Regime 2         │    │ Regime 3         │
 │ resident GPU  │             │ RAM → GPU        │    │ disk → GPU       │
 │ pre-encode    │             │ stream a chunk,  │    │ stream a chunk,  │
 │ once; batch = │             │ encode on the fly│    │ encode on the fly│
 │ on-GPU index  │             │ (pinned on CUDA) │    │ (memmap read)    │
 └───────────────┘             └──────────────────┘    └──────────────────┘
  no transfer or                re-stream the subset once per epoch, in VRAM-sized
  re-encode, ever               chunks (load = bs · batches_per_load) — per chunk,
                                not per minibatch
```

**7. Train** (`training.py`). `run_emulator` builds the model, optimizer, and
scheduler from spec dicts, runs the per-epoch loop (annealed robustness, a
validation pass each epoch, keeping the best epoch by `f(dchi2 > 0.2)` (the
`frac>0.2` of the logs)), and
returns the histories. When an `ema:` block is set the selection and reported
metrics run on a Polyak weight average (the scheduler still steps on the raw
median), and the shipped model is the best average.

`experiment.py` (`EmulatorExperiment`) ties steps 1–7 into one object, so each
driver is a thin wrapper that varies one knob:

```
                       EmulatorExperiment   (steps 1–7)
                                │
     ┌────────────┬─────────────┼───────────────┬──────────────────┐
     ▼            ▼             ▼               ▼                  ▼
train         tune          sweep_ntrain  sweep_hyperparam  bakeoff_activation
  one run    Optuna search f(dchi2) vs N  one YAML knob      one curve per act
             (multi-GPU,   (multi-GPU,    (multi-GPU,        (multi-GPU, by act)
              journal)      LPT, --gpu-    even split,
                            pack)          --gpu-pack)
```

---

## 20. Appendix: the chi2 metric (Mahalanobis)

The loss and the reported metric are both a **chi2**, which is a squared
**Mahalanobis distance** — the distance between two points measured *in units of
the data's own spread and correlations*, not in raw coordinate units. For a
residual `r = pred − truth`, the squared Mahalanobis distance is

```
   d²  =  rᵀ · C⁻¹ · r          C = data covariance,  C⁻¹ = precision (Cinv)
```

and that *is* the chi2. The contrast with plain distance is the whole point:

| | formula | what it does |
|---|---|---|
| plain Euclidean | `rᵀ r = Σ rᵢ²` | every entry counts equally |
| **Mahalanobis** | `rᵀ C⁻¹ r` | divide each direction by its variance, remove correlations → "how many **σ** off", not "how many raw units off" |

Two limits make it concrete:

- **Diagonal `C`** (variances σᵢ², no correlations): `d² = Σ (rᵢ / σᵢ)²` — just
  z-scores squared. A 1-unit error on a tight bin (small σ) costs far more than on
  a loose bin.
- **`C = I`**: collapses back to plain Euclidean.

The tie to this codebase: **whitening is exactly the coordinate change that turns
Mahalanobis into Euclidean.** In the whitened basis the covariance becomes the
identity, so

```
   rᵀ · C⁻¹ · r   =   ‖ whiten(r) ‖²
```

That is the "whiten the output, keep the metric" line from step 3: the network
trains on a clean `‖·‖²` in the whitened basis, while the loss still scores the
true correlated metric — it un-whitens the residual and contracts the masked
`Cinv`. For the standard full whitening the two are equal to rounding; the
diagonal-whitening variant (`DiagonalGeometry`, used for the CNN) breaks that
equality, which is why the loss always keeps the explicit `Cinv` contraction
rather than collapsing to a mean squared error.

**Mnemonic:** Mahalanobis = Euclidean distance *after whitening* (distance in σ
units, with correlations removed).

---

## 21. Appendix: activation functions

The `ResBlock` nonlinearity is a **learnable, per-feature activation**: every
feature (one entry of the vector) carries its own shape parameters, trained with
the network. The default is the paper's $H(x)$; three generalizations are
available, selected by name (`--activation`, or `make_activation` in
[`activations.py`](emulator/activations.py)). Throughout, $\odot$ is the
elementwise (Hadamard) product, $\sigma(z) = 1/(1 + e^{-z})$ is the logistic
sigmoid, and each Greek symbol is a length-`dim` vector (one value per feature).

### The paper's $H(x)$

From eq. (6) of [arXiv:2505.22574](https://arxiv.org/pdf/2505.22574):

$$H(x) = \Big( \gamma + (1 + e^{-\beta \odot x})^{-1} \odot (1 - \gamma) \Big) \odot x$$

Because $(1 + e^{-\beta x})^{-1} = \sigma(\beta x)$, this is a per-feature
**interpolation between the identity and a Swish gate**:

$$H(x) = \gamma \odot x + (1 - \gamma) \odot \mathrm{Swish}_\beta(x), \qquad \mathrm{Swish}_\beta(x) = x \sigma(\beta x).$$

The gate $\gamma + (1-\gamma) \sigma(\beta x)$ runs from $\gamma$ (as
$x \to -\infty$) to $1$ (as $x \to +\infty$), so $H$ is **asymptotically linear
on both tails** — slope $\gamma$ on the left, $1$ on the right. That
non-saturation is why it beats $\tanh$ (whose slope vanishes) for these
emulators. $\gamma$ sets the left-tail slope (the linear-vs-nonlinear mix),
$\beta$ the kink sharpness near $x = 0$. At init $\gamma = \beta = 0$, so $H$
starts as $0.5 x$ (since $\sigma(0) = 0.5$) and training shapes each feature's
curve. Two learnable vectors per feature ($\gamma$, $\beta$).

### Generalizations

Each is a strict superset of $H$ and recovers it at initialization. They give the
activation more freedom where the target is hard; $H$ stays the default.

**Multi-gate** (`multigate`, `GatedActivation`) — replace the single Swish gate
with a sum of $K$ sigmoids, a learnable slope-vs-$x$ schedule in the bulk:

$$\mathrm{gate}(x) = a_0 + \sum_{k=1}^{K} w_k \sigma\big(\beta_k (x - \mu_k)\big), \qquad \mathrm{out} = \mathrm{gate}(x) \odot x.$$

Every term is a bounded sigmoid times $x$, so the output stays asymptotically
linear (slope $a_0$ to the left, $a_0 + \sum_k w_k$ to the right). $H$ is the
$K = 1$ case ($a_0 = \gamma$, $w_1 = 1 - \gamma$, $\mu_1 = 0$); the general form
also frees the right-tail slope and the kink center $\mu$. $3K + 1$ vectors per
feature.

**Bounded power tail** (`power`, `PowerGatedActivation`) — keep $H$'s gate but
apply it to a signed power transform $\psi_p$, linear near $0$ and $\sim |x|^p$
in the tail, with $p$ learnable and boxed into $[p_{\min}, p_{\max}]$ (default
$[0.5, 1.5]$, between $\sqrt{x}$ and $x^{1.5}$):

$$\psi_p(x) = \mathrm{sign}(x) \frac{(1 + |x|)^p - 1}{p}, \qquad p = p_{\min} + (p_{\max} - p_{\min}) \sigma(\rho),$$

$$\mathrm{out} = \big(\gamma + (1 - \gamma) \sigma(\beta x)\big) \odot \psi_p(x).$$

The $/p$ normalization gives $\psi_p$ slope $1$ at $x = 0$ for **any** $p$, so
$p$ reshapes only the tail; the base $1 + |x| \ge 1$ keeps any real $p$ finite
(no `NaN`), and the sigmoid box prevents a runaway exponent — safe
superlinearity, unlike a raw $x^n$. $p = 1$ (at $\rho = 0$) recovers $H$. Three
vectors per feature ($\gamma$, $\beta$, $\rho$).

**Both** (`gated_power`, `GatedPowerActivation`) — the multi-gate bulk times the
bounded power tail:

$$\mathrm{out} = \Big(a_0 + \sum_{k=1}^{K} w_k \sigma\big(\beta_k(x - \mu_k)\big)\Big) \odot \psi_p(x).$$

$3K + 2$ vectors per feature.

### Selecting one

| `--activation` | class | adds over `H` | params / feature |
|---|---|---|---|
| `H` (default) | `activation_fcn` | — (the paper's gate) | 2 |
| `multigate` | `GatedActivation` | K-sigmoid bulk slope schedule | 3K + 1 |
| `power` | `PowerGatedActivation` | bounded learnable tail exponent | 3 |
| `gated_power` | `GatedPowerActivation` | both of the above | 3K + 2 |
| `relu` | `nn.ReLU` | parameter-free (a plain rectifier) | 0 |
| `tanh` | `nn.Tanh` | parameter-free; saturates (pair with `norm: per_feature`) | 0 |

$K$ (the gate count for the multi-gate families) is `make_activation`'s
`n_gates`, default 3. `relu` and `tanh` are the two parameter-free
(non-learnable) baselines — a plain rectifier and the classic saturating
tanh; the paper's affine `norm` (`per_feature`) is `tanh`'s saturation
guard.

---

## 22. Appendix: precedence — who wins when settings collide

Configuration arrives from several places — the YAML, the driver flags, the
per-phase override blocks, and the built-in defaults. When two of them speak
to the same setting, these tables say which one the run actually uses. Every
row is the resolved, consumed behavior (the startup `print_design` banner
shows it for a given run).

### A. Activation family

Two slots: the shared slot (the trunk and every component that sets no
activation of its own) and the head slot (a `rescnn` / `restrf` correction
head). The shared slot resolves `--activation` flag > `model.activation` >
`H`; the head slot follows the shared slot unless `model.cnn`/`.trf.activation`
pins it.

| `--activation` | `model.activation` | `model.<head>.activation` | trunk gets | head gets |
|---|---|---|---|---|
| absent | `H` | absent | `H` | `H` (shared) |
| absent | `H` | `gated_power` | `H` | `gated_power` (pin) |
| `power` | `H` | absent | `power` | `power` (flag wins the shared slot) |
| `power` | `H` | `gated_power` | `power` | `gated_power` (pin holds; a startup warning prints) |

Why: most-specific wins — an explicit head pin is the most specific
statement, so it holds even against the flag; the flag and `model.activation`
only ever set the shared slot.

Four rules ride with this table:
- The flag sets only the type; `n_gates` is always read from the YAML
  (`model.activation.n_gates` for the shared slot, the head block's own
  `n_gates` for a pin).
- The bake-off and a `model.activation` sweep rewrite the shared slot per
  curve; a pinned head stays fixed across every curve (the driver prints the
  warning once at startup).
- Spellings: on a two-phase model `head: activation:` is a second spelling of
  the pin (canonical is `model.cnn`/`.trf.activation`, which also reads on
  single-phase YAMLs); giving both spellings is an error (keep one, even when
  they agree). `trunk: activation:` is an error (section B).
- License: a per-head pin (either spelling) needs a frozen-trunk head phase —
  `trunk_epochs > 0` and `freeze_trunk` true. When the trunk and head train
  together (`freeze_trunk: false`, or no two-phase schedule) the network keeps
  one family and the pin errors with a teaching message.

### B. Phase blocks vs the top level (two-phase models)

Per key, a `trunk:` / `head:` value beats the top-level `train_args`, which
beats the built-in default — but the override semantics differ by key:

| key | override semantics | note |
|---|---|---|
| `lr` | overlay `{lr_base, warmup_epochs}` onto the top-level `lr` | `bs_base` is run-global: inside a phase `lr` it is an error |
| `scheduler` | full kwargs replacement | the scheduler class stays the run's: a `cls` in a phase is an error |
| `loss` | full block replacement | `{mode, berhu}`; no merge with the top-level block |
| `trim` / `focus` | full block replacement | restart at the phase's own epoch 1 |
| `clip` / `rewind` | value replacement | |
| `ema` | full block replacement | `ema: null` (key present, empty value) is an explicit per-phase off, overriding an inherited top-level `ema` |
| `activation` | head only: an alias for `model.<head>.activation`, consumed when the model is built rather than during training | Legal in `head:` because the head trains only in phase 2, so the pin has one owner. An error in `trunk:` — the trunk is the same modules in both phases, so a phase-local trunk activation cannot exist, and the error says so. |

Why: phase blocks are diffs against the top level, not containers (the
`bs_base` rule); construction knobs are run-global, with the one user-ruled
exception `head: activation:`, coherent only because the head component and the
head phase coincide.

### C. Single-phase demotion (the same YAML on `resmlp`)

A model without `set_train_phase` (any `resmlp`, including the IA `nla` /
`tatt`) has no phases, so `resolve_phase_args` demotes the schedule keys before
training:

| key | on a single-phase model |
|---|---|
| `trunk:` | merged into the top level (full-replace per key; `lr` overlays, keeping `bs_base`) — the trunk is just the global objective |
| `head:` | dropped |
| `trunk_epochs` | dropped |
| `freeze_trunk` | dropped |

Why: one shared YAML drives both model families; a quiet one-line notice names
what was demoted, so the banner still tells the truth about the run.

### C2. Two-phase schedule modes

| `trunk_epochs` | `freeze_trunk` | schedule |
|---|---|---|
| `0` / absent | (must be absent) | joint training from epoch 1 |
| `N > 0` | `true` / absent | trunk-only for `N` epochs, then a frozen trunk + head-only (the default) |
| `N > 0` | `false` | trunk-only for `N` epochs, then trunk + head together (a joint fine-tune) |

Why: `freeze_trunk` only means something with a two-phase schedule; setting it
without `trunk_epochs > 0` is an error (it would silently do nothing). The
`head:` block configures phase 2 in either mode.

### D. Loss-block spellings

| config | result |
|---|---|
| `loss` absent, or no `mode` key | `mode: sqrt` |
| a berhu mode with no `berhu:` sub-block | knots `{knot: 0.2, cap: 10}` |
| the knot block spelled `berhu:` (the family; sweep-safe) | accepted |
| the knot block spelled after the active mode (`berhu_capped:` under `mode: berhu_capped`) | accepted |
| both spellings present | an error — no silent winner, ever |

Why: one setting, one home; the family spelling `berhu:` survives a `mode`
sweep, the mode-named spelling reads literally, and giving both is ambiguous.

### E. Sweeps and searches

| context | rule |
|---|---|
| `sweep_hyperparam` | the sweep leaf beats the `train_args` baseline (a deep copy per point) |
| sweep parameter `model.activation` | a special case: it sets the experiment's shared slot, so `--activation` must be unset |
| `model.name` / `model.ia` | refused as sweep axes (they change the model class) |
| a phase axis (`head.*` / `trunk_epochs` / `freeze_trunk` / `trunk.*`) on a single-phase model | refused at startup (`validate_sweep_paths`) |
| tune ranges `[default, min, max, kind]` | the suggested value per trial beats the default; the default is what the train drivers use and what warm-starts trial 0 |
| `sweep_ntrain` | the driver's per-point `n_train` beats `data.n_train` (the `stage_train` argument) |
| the top-level `pce:` block | not a `train_args` leaf, so structurally unsweepable / unsearchable — one base per study; a pce knob under `train_args` would sweep without refitting the base (a silent no-op) |

Why: a sweep varies exactly one concretized leaf per point over the shared
baseline; axes that would be silently dropped, or that change the class, are
refused up front.

### F. Constructor / driver args vs the YAML

| setting | precedence |
|---|---|
| `device` | an explicit arg beats auto-detect (CUDA > MPS > CPU) |
| `thresholds` | a constructor arg beats `DEFAULT_THRESHOLDS` |
| `rescale` | a driver flag only (no YAML key) |
| `pce` (the top-level block) | exclusive with `rescale` and `model.ia` — each replaces the chi2 loss, so `validate_pce` errors on either combination (use one at a time) |
| `ram_frac` | the parallel `sweep_ntrain` / `sweep_hyperparam` / bake-off workers force `0` (stream from the one shared dump memmap, no private copy); the tune (Optuna) workers instead divide `data.ram_frac` by the worker count (each stages its own subset concurrently); a serial run uses `data.ram_frac` (default 0.7) |
| `compile_mode` | `model.compile_mode` (YAML) beats the architecture default — `"default"` for the conv / TRF heads (reduce-overhead's CUDA-graph capture trips on the gated skip-add), `"reduce-overhead"` for `resmlp` |

Why: the drivers own the runtime placement and the multi-worker memory budget;
the YAML owns the per-model knobs, with a sensible default when a key is
omitted.

### G. Deliberately no knob (nothing to win)

| setting | why there is no override |
|---|---|
| evaluation batch size | derived from `n_val` (a ~1024-row target, minimal tail pad), decoupled from the training `bs` by design |
| `bs_base` | the run-global sqrt-rule anchor; a phase-level value is an error |
| optimizer / scheduler class | fixed (`AdamW` / `ReduceLROnPlateau`) in the drivers; a phase scheduler overrides only the kwargs |
| TRF token width | pinned to the padded bin length (physics: no embedding, no adapters) |
| TRF MLP interior width | pinned to the token width (`n_mlp_blocks` is depth only; the width knob is shelved) |

Why: these are fixed by design or derived from the data, so there is no second
source to disagree with — the heads-up is that a "missing knob" is intentional.

---

## 23. Appendix: scripting a saved emulator (without Cobaya)

A saved emulator has two doors. In a chain, the Cobaya theory blocks
serve it — `emul_cosmic_shear` for data vectors ([section 1](#run-the-saved-emulator-in-a-cobaya-mcmc)),
`emul_scalars` for derived parameters ([section 14](#14-scalar-derived-parameter-emulators)),
`emul_cmb` for CMB spectra ([section 15](#15-emulating-cmb-spectra-tt--te--ee--phi-phi)),
`emul_baosn` for the expansion history ([section 16](#16-emulating-the-expansion-history-hz-bao-and-sn-distances)),
and `emul_mps` for the matter power spectrum ([section 17](#17-emulating-the-matter-power-spectrum-hybrid-inference-emul2)).
Everywhere else — profile likelihoods, quick checks, plots, batch
evaluation — you call the same artifact directly from Python, with no
Cobaya anywhere:

```
                        ┌── in a chain: the Cobaya theory blocks
saved emulator  ────────┤     emul_cosmic_shear / emul_scalars /
 <root>.h5 + .emul      │     emul_cmb / emul_baosn / emul_mps
                        └── in a script: EmulatorPredictor
                              predict(dict) -> the family's native
                                              output (table below)
```

Both doors share one decode path, so a script and a chain get identical
values from the same artifact by construction.

### Load one

```python
import os, sys
sys.path.insert(0, os.path.join(os.environ["ROOTDIR"],
                                "external_modules/code/emulators_code_v2"))
from emulator.inference import EmulatorPredictor

pred = EmulatorPredictor(
    os.path.join(os.environ["ROOTDIR"],
                 "projects/lsst_y1/chains/emulator_resmlp_t256_ntrain250000"),
    device="cuda")          # "cpu" and "mps" work too
```

The one argument that matters is the path root — the saved emulator
without its extension, resolving to `<root>.h5` + `<root>.emul`. The
input names, their order, the rescalings, and the outputs are all read
from the file; there is nothing else to configure.

### What `predict` returns

| the artifact | `predict({...})` returns |
|---|---|
| a data-vector emulator (sections 2–13) | a 1-D numpy array: by default this emulator's own probe block; build with `dv_return="3x2pt"` to get the full-length vector instead, kept entries in place and zeros elsewhere |
| a scalar emulator (section 14) | a `{name: value}` dict of plain Python floats, one entry per emulated output |
| a CMB emulator (section 15) | a 1-D numpy array of C_ell on the training multipole grid, in the dump units (muK^2; phi-phi dimensionless). The stored amplitude law is already multiplied back — the values are physical C_ell |
| a background emulator (section 16) | a dict `{"z": the stored grid, "<quantity>": values}` — e.g. `"Hubble"` in km/s/Mpc; feed it to `distance_interpolators` for the distances (pattern below) |
| a matter-power emulator (section 17) | a dict `{"z": ..., "k": ..., "<quantity>": an (nz, nk) surface}` in LAW space — for a syren-law artifact this is log(P / P_syren); the syren base is multiplied back by `emul_mps`, not here |

The input is a dict of physical parameter values. Its keys are the
artifact's stored input names — read them off `pred.names`; a scalar
artifact also exposes `pred.output_names`. Extra keys in the dict are
ignored; a missing one is an error naming it.

### The profile-script pattern

The legacy per-emulator classes took `file` / `extra` / `ord` /
`extrapar` lists and one hardcoded getter per output. The v2 form of
the same computation:

```python
etheta = EmulatorPredictor(root_thetaH0, device="cuda")
erd    = EmulatorPredictor(root_rdrag,   device="cuda")

out = etheta.predict({"omegabh2": 0.02238,
                      "omegach2": 0.1201,
                      "thetastar": 1.04109})
h0, om = out["H0"], out["omegam"]
rd     = erd.predict({"omegabh2": 0.02238,
                      "omegach2": 0.1201})["rdrag"]
```

Each call evaluates one point; loop for a profile's grid. The
`compile_model` flag defaults to False — single-point latency rarely
pays back a compile.

### The background pattern

A background artifact returns its quantity on the stored redshift
grid; the distances come from the same function the Cobaya adapter
uses, so a script and a chain share one convention:

```python
from emulator.background import distance_interpolators

ehub = EmulatorPredictor(root_hubble, device="cuda")
out  = ehub.predict({"omegabh2": 0.02238, "omegach2": 0.1201,
                     "H0": 67.36, "w": -1.0})
dist = distance_interpolators(z_grid=out["z"], h_grid=out["Hubble"])
dist["dl"](1.5)      # luminosity distance at z = 1.5, in Mpc
dist["H"](0.5)       # H(z = 0.5), km/s/Mpc
```

### One caveat

The predictor reads schema-v2 artifacts only. The legacy `.joblib` /
`.pt` files with their `extra` / `ord` sidecars stay on the legacy
classes; the v2 replacement is a retrain
([section 14](#14-scalar-derived-parameter-emulators) — scalar maps
retrain from a chain in minutes).

---

## 24. AI-Usage

AI Usage: This library (under the `dev` folder) was developed with Claude
Code assistance. However, Prof. Miranda heavily influenced the code at every
level, from macro-designed implementation and changes to minute Python
choices.
