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
[chi2 metric](#13-appendix-the-chi2-metric-mahalanobis)).

One line: raw dumps → stage → whiten params (input) and data vector (output) →
ResMLP / ResCNN / ResTRF → chi2 loss → train. `EmulatorExperiment` wires it together; each
driver varies one thing (one run, a tune, an `N_train` sweep, an activation bake-off).

The code map — how the package is laid out, what each file does, and where to
edit for a given change — lives in [`emulator/README.md`](emulator/README.md).

## Contents

- **Code map** (layout, file roles, change X → edit Y): [`emulator/README.md`](emulator/README.md)

1. [Run it](#1-run-it)
    1. [The `sweep:` block (one-knob sweeps)](#sweep-block)
    2. [Multi-GPU execution and packing](#multi-gpu)
2. [The YAML file](#2-the-yaml-file)
3. [`data`](#3-data)
4. [Training globals](#4-training-globals)
5. [`loss`](#5-loss)
6. [optimizer, lr, scheduler](#6-optimizer-lr-scheduler)
7. [`trim`](#7-trim)
8. [`focus`](#8-focus)
9. [`ema`](#9-ema)
10. [`model`](#10-model)
11. [Two-phase schedule + the `trunk:` / `head:` blocks](#11-two-phase-schedule--the-trunk--head-blocks)
12. [Appendix: the pipeline](#12-appendix-the-pipeline)
13. [Appendix: the chi2 metric (Mahalanobis)](#13-appendix-the-chi2-metric-mahalanobis)
14. [Appendix: activation functions](#14-appendix-activation-functions)
15. [Appendix: precedence — who wins when settings collide](#15-appendix-precedence--who-wins-when-settings-collide)
16. [AI-Usage](#16-ai-usage)

---

## 1. Run it

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
D=external_modules/code/emulators/emultrf/dev
```

One training run — train the YAML's model once; `--diagnostic` adds a
multipage PDF of accuracy diagnostics:

```bash
python $D/train_single_emulator_cosmic_shear.py \
  --root projects/lsst_y1/ --fileroot emulators/training_scripts/ \
  --yaml train_single_emulator_cosmic_shear.yaml --diagnostic diagnostic
```

The `N_train` learning curve — how does accuracy improve as the training set
grows? This driver retrains the same model at several training-set sizes and
plots the error metric (`frac>0.2`, defined in [section 2](#2-the-yaml-file))
against N: a curve still falling at the largest N says more data will help; a
flat tail says the model, not the data, is the limit. Sweep points are
independent trainings, so they run in parallel — one whole training per GPU,
all visible GPUs by default ([Multi-GPU](#multi-gpu) below):

```bash
python $D/sweep_ntrain_emulator_cosmic_shear.py \
  --root projects/lsst_y1/ --fileroot emulators/training_scripts/ \
  --yaml train_single_emulator_cosmic_shear.yaml --n-points 8 --out curve
```

A one-knob sweep — one full training per value of a single YAML-chosen knob
(learning rate, kernel size, batch size, ...), to see that knob's effect in
isolation; the knob and values live in the [`sweep:` block](#sweep-block):

```bash
python $D/sweep_hyperparam_emulator_cosmic_shear.py \
  --root projects/lsst_y1/ --fileroot emulators/training_scripts/ \
  --yaml train_single_emulator_cosmic_shear.yaml --out lrsweep
```

A hyperparameter search — Optuna (https://optuna.org) is a search library: it
proposes trial settings, watches the results, and concentrates new trials
where the metric improves. The searched ranges are the YAML's
`[default, min, max, kind]` leaves; `--n-trials` bounds the study:

```bash
python $D/tune_single_emulator_cosmic_shear.py \
  --root projects/lsst_y1/ --fileroot emulators/training_scripts/ \
  --yaml tune_single_emulator_cosmic_shear.yaml --n-trials 64
```

The activation bake-off — trains the same model once per activation family
over a grid of training sizes and overlays their learning curves: a
head-to-head showing whether a family genuinely learns faster (a lower curve
everywhere) or just ties:

```bash
python $D/bakeoff_activation_emulator_cosmic_shear.py \
  --root projects/lsst_y1/ --fileroot emulators/training_scripts/ \
  --yaml train_single_emulator_cosmic_shear.yaml --out bakeoff
```

On a card with far more memory than one training needs (an H200), add
`--gpu-pack` to either sweep: points estimated at ≤ 20% of the GPU run four
to a card, ≤ 40% two to a card, bigger ones exclusive (off by default — on a
12 GB RTX 3060 one training is the card). The details live in
[Multi-GPU execution and packing](#multi-gpu) below.

The YAML has two top-level blocks — `data` and `train_args`. The next
chapter ([The YAML file](#2-the-yaml-file), sections 2–11) documents every
block with its math, options, and a small example; templates live in
`example_yamls/` (one per driver style — copy one into your `--fileroot` and
edit it). The `sweep:` block is documented [below](#sweep-block).

### The `sweep:` block (one-knob sweeps) <a name="sweep-block"></a>

`sweep_hyperparam_emulator_cosmic_shear.py` reads one extra top-level YAML
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

Outputs under `--fileroot`: `<--out>.txt` (`save_sweep_table`: numeric values
as a value/frac table; categorical or boolean values as an index/frac table
with a `# values: 0=…, 1=…` label line — `np.loadtxt` reads either) and
`<--out>.pdf` (`plot_sweep_curve`). The full template is
`example_yamls/sweep_hyperparam_emulator_cosmic_shear.yaml`, with the common
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
| `tune_single` | Optuna trials | one worker per GPU, one shared study | `--journal` |

**`--gpu-pack` (both sweep drivers; off by default)** co-locates small
trainings on one card: a point's estimated VRAM share sets its token cost
(≤ 20% → 4 per GPU, ≤ 40% → 2, larger runs exclusive), so launch-bound runs
share a card. Use it on big cards with small-to-mid `N_train`, not on small
cards or when per-epoch timings must stay comparable; the token math lives in
`scheduling.py` (`estimate_train_vram_fraction`, `vram_tokens`).

**Parallel Optuna (`tune_single --n-gpus N`)** shares a single study through a
journal file (`--journal`): the parent enqueues the warm-start, `--n-trials`
splits across workers, and reusing the journal resumes it (serial on 1 GPU/MPS).

---

## 2. The YAML file

Two top-level blocks: `data` (where the training vectors come from and how
many) and `train_args` (the whole run — objective, optimizer, schedules,
model). Any numeric leaf may be a scalar (the train drivers use it) or a
`[default, min, max, kind]` search range — `kind` is `int` | `float` | `log`
(`tune_single` searches it, the others collapse it to the default). Sections
3–11 document each block; the
collision rules (which source wins when two set the same thing) live in the
[precedence appendix](#15-appendix-precedence--who-wins-when-settings-collide),
templates in `example_yamls/`, and the `sweep:` block in [Run it](#sweep-block).

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

Six terms the chapter uses (details: appendices
[12](#12-appendix-the-pipeline)–[13](#13-appendix-the-chi2-metric-mahalanobis)):

- data vector (dv): the masked cosmic-shear two-point functions xi+/- stacked
  into one vector — what the network predicts.
- chi2: prediction error measured in the analysis covariance, `r^T Cinv r`
  ([appendix 13](#13-appendix-the-chi2-metric-mahalanobis)). The headline
  metric, written `frac>0.2` in the logs: the fraction of validation
  cosmologies with delta-chi2 above 0.2 — the goal is to drive it down.
- whitened: rotated and rescaled so the components are decorrelated with unit
  variance — the form the network sees, input and output
  ([appendix 12](#12-appendix-the-pipeline)).
- theta order: the data vector re-sorted to vary smoothly along the angular
  axis — the basis the correction heads work in.
- trunk / head: every architecture is a shared ResMLP trunk; `rescnn` /
  `restrf` add a gated correction head on top ([section 10](#10-model)).
- dump: the big on-disk (params, dv) table the physics code wrote; training
  memmaps it (reads slices from disk, never the whole file) and stages only
  the rows it needs ([section 3](#3-data)).

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
may fill before it streams from the disk memmap instead.

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
| `omegabh2_lo/_hi` | $\Omega_b h^2$ | $\Omega_b\,(H_0/100)^2$ | 0.0224 |
| `omegam2h2_lo/_hi` | $\Omega_m^2 h^2$ | $(\Omega_m\,H_0/100)^2$ | 0.045 |
| `omegamh2_lo/_hi` | $\Omega_m h^2$ | $\Omega_m\,(H_0/100)^2$ | 0.143 |
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

The run-level knobs that are not their own block.

- `nepochs` — passes over the training set.
- `bs` — the training minibatch. The validation pass uses a **derived**
  batch (a ~1024-row target, `derive_eval_bs`), independent of `bs`, so a
  small `bs` does not slow scoring.
- `trunk_epochs` / `freeze_trunk` — the two-phase schedule (section 11); the
  mode table is precedence
  [C2](#15-appendix-precedence--who-wins-when-settings-collide).
- `silent` — suppress the per-epoch progress lines.
- `clip` — a per-step gradient-norm ceiling (0 = off); the full gradient is
  rescaled toward the ceiling, keeping its direction, so one monster-outlier
  batch (a batch with one extreme sample) cannot kick the weights:

$$g \leftarrow g \cdot \min\!\left(1,\ \frac{\mathrm{clip}}{\lVert g \rVert}\right)$$

- `rewind` — on every plateau lr cut (the scheduler's, [section 6](#6-optimizer-lr-scheduler)), reload the best weights + optimizer
  snapshot (keeping the reduced lr), bounding an excursion into a bad basin to
  at most `patience` epochs.

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
([Mahalanobis](#13-appendix-the-chi2-metric-mahalanobis)); the batch loss is
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

$$L(c) = \begin{cases} \sqrt{c} & c \le k \\[4pt]
\dfrac{c+k}{2\sqrt{k}} & c > k \end{cases}$$

and `berhu_capped` adds $\dfrac{2\sqrt{Kc}+k-K}{2\sqrt{k}}$ for $c > K$.

$C^1$ at every knot. $k = $ `berhu.knot` (default 0.2, the frac>0.2 goal),
$K = $ `berhu.cap` (default 10). This is textbook BerHu in the whitened
residual norm with $\delta = \sqrt{k}$ — the knots are in chi2 units, applied
per sample (the Mahalanobis aggregate). Vote intuition: `sqrt` gives every
sample an equal vote; `chi2`'s vote grows with $c$ (the tail dominates);
`berhu` keeps equal bulk votes and rises ~×7 across $(k, K)$; `berhu_capped`
plateaus above $K$ so a chi2=100 monster stays bounded.

The `berhu:` sub-block sets the knots (spell it `berhu:` — the family, so it
survives a `mode` sweep — or after the active mode as `berhu_capped:`; giving
both is an error, see precedence
[D](#15-appendix-precedence--who-wins-when-settings-collide)). An optional
`anneal:` (presence = on) starts as plain sqrt and blends into the berhu shape
on the [shared schedule](#7-trim), $s: 0 \to 1$:

$$L_s = (1-s)\,\sqrt{c} + s\,L(c)$$

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

- `optimizer` — the class is fixed to **AdamW**; `weight_decay` decays only
  the true weight matrices (the `.weight` of `Linear` / `Conv1d` / `BinLinear`),
  never biases, norms, or activation parameters, and runs fused on CUDA (a
  faster fused kernel). The weight-decay rule is detailed below.
- `lr` — the learning rate follows the sqrt-noise rule off a batch anchor
  (bigger batches average away gradient noise, so the step can grow with
  sqrt(bs)), then a linear warmup:

$$\mathrm{lr} = \ell\,\sqrt{B/B_0}$$

  with $\ell$ = `lr_base`, $B$ = `bs`, $B_0$ = `bs_base`.
  `bs_base` is the run-global anchor (never inside a phase block);
  `warmup_epochs` linearly ramps the lr from 0 over the first epochs.
- `scheduler` — the class is fixed to **ReduceLROnPlateau**; `{mode, patience,
  factor}` are its kwargs, stepped every epoch on the **raw** validation
  median (the EMA average never feeds it). A per-phase `scheduler:` replaces
  the kwargs but keeps the class (precedence
  [B](#15-appendix-precedence--who-wins-when-settings-collide)).

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

$$w \leftarrow w - \mathrm{lr}\,\lambda\,w$$

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

$$\bar\theta \leftarrow \beta\,\bar\theta + (1-\beta)\,\theta
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

Every architecture is a shared ResMLP trunk; `rescnn` / `restrf` add a gated
correction head in theta order (the gate is a learnable scalar scaling the
correction, near 0 at init, so they start as the trunk):

```
params ─▶ ResMLP trunk ─▶ y (full-whitened)
                          │  basis change to theta order
                          ▼
                    head blocks (cnn conv | trf attention)
                          │
          y + gate · correction ─▶ whitened data vector
```

Intrinsic alignments (IA): galaxies have correlated intrinsic shapes
independent of lensing — the main astrophysical contaminant of cosmic shear;
the analysis models it with amplitude parameters. An `ia` design makes the net
emit whitened templates that a closed-form polynomial combines, so the IA
amplitudes never enter the network (the emulator is exact in them). For `nla`
(3 templates, amplitude $A_1$):

$$\xi = K_0 + A_1 K_1 + A_1^2 K_2$$

For `tatt` (10 templates; amplitudes $a_1, a_2, b_{TA}$ — the
IA field is $a_1 O_1 + a_2 O_2 + a_1 b_{TA} O_{1\delta}$, so
$\xi$ is quadratic in it: 1 GG + 3 GI + 6 II terms):

$$\xi = K_0 + a_1 K_1 + a_2 K_2 + a_1 b_{TA} K_3
+ a_1^2 K_4 + a_2^2 K_5 + (a_1 b_{TA})^2 K_6
+ a_1 a_2 K_7 + a_1^2 b_{TA} K_8 + a_1 a_2 b_{TA} K_9$$

### `mlp`

The trunk (required — every architecture is built on it): `width`, `n_blocks`.

```yaml
  mlp:
    width:    128
    n_blocks: 4
```

### `activation`

The activation family, `{type, n_gates}` or a bare type string: the four
learnable families `H` / `power` / `multigate` / `gated_power` plus the
parameter-free `relu` / `tanh` (pair `tanh` with `norm: per_feature`, the
saturation guard); their math is the
[activation appendix](#14-appendix-activation-functions). This sets the shared
family (trunk + default). A `rescnn` / `restrf` head may pin its own with
`model.cnn`/`.trf.activation` (absent = share the trunk's); the pin needs a
frozen-trunk head phase, `head: activation:` is its alias, and the precedence
+ warning are precedence
[A](#15-appendix-precedence--who-wins-when-settings-collide).

```yaml
  activation:
    type:    H
    n_gates: 3
```

### `norm`

The ResBlock normalization slot, applied inside the trunk before each
activation — the paper's saturation guard. One of three:

- `affine` (default; absent = this) — the paper's per-layer $g x + b$,
  one scalar gain/bias pair per layer (`Affine`); byte-identical to the
  model today.
- `per_feature` — a width-long gain/bias, one pair per feature
  (`FeatureAffine`); the escalation when one scalar pair cannot hold
  every unit's operating point. Pair it with `tanh`.
- `none` — `nn.Identity`, no normalization (an ablation).

```yaml
  norm: affine    # affine (default) | per_feature | none
```

Only the trunk ResBlocks read it — the transformer head's internal
LayerNorm and the conv head have no norm slot.

**Why no batchnorm.** Batch normalization is deliberately not offered:
its batch coupling would confound the batch-size and EMA experiments,
and its train/eval running-stats split — with buffers outside the EMA
weight average — risks baking a fixed mode under the compiled eval twin.
The paper prescribes the affine as batchnorm's replacement, and
`per_feature` is the escalation.

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
the same sense in which H's gamma / beta are per-feature.

| | `affine` (per layer — the paper's) | `per_feature` |
|---|---|---|
| parameters | one pair $(g, b)$ for the whole layer — 2 | vectors $g_i, b_i$ — 2·width (256 at width 128) |
| action | $g \cdot x + b$ broadcast over all features | $g_i x_i + b_i$, each feature its own |
| what it can fix | a global scale drift | each unit's individual operating point |

### `cnn` (name `rescnn`)

| knob | what |
|---|---|
| `kernel_size` | conv kernel width (odd), tuned as if one block |
| `rescale_kernel` | shrink the per-block kernel with depth at a fixed receptive field |
| `groups` | channel-mixing cuts (`2` = xi+/xi- split; `3` / `6` on the factored head) |
| `separable` | factor each block into a depthwise + pointwise conv |
| `film` | re-inject the parameters as an identity-init per-channel affine (cosmology-aware) |
| `n_blocks` | stacked conv + activation blocks |
| `gate_init` | initial correction-gate scale (small, not 0) |
| `activation` | the head's own family (above) |

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

| knob | what |
|---|---|
| `n_heads` | attention heads; must divide the token width (26 → 1 \| 2 \| 13) |
| `n_blocks` | stacked transformer blocks |
| `n_mlp_blocks` | per-token MLP depth; every layer at the token width, no width knob (depth only) |
| `shared_mlp` | one MLP for all tokens (the textbook block) vs the per-token default |
| `film` | cosmology-aware per-token affine (as `cnn.film`) |
| `gate_init` | initial correction-gate scale |
| `activation` | the head's own family (above) |

```
one token = one bin, width 26            n_heads: 2 -> d_head = 13

    ┌──────────────┬───────────────┐
    │ head 1: 1-13 │ head 2: 14-26 │     the feature axis is sliced
    └──────┬───────┴───────┬───────┘     into n_heads equal parts:
           │               │             26 = n_heads x d_head, so
    G x G attention  G x G attention     n_heads must divide 26
    over all bins,   over all bins,      (1 | 2 | 13)
    using slice 1    using slice 2
           │               │
           └─── concat ────┴──▶ width 26 again (wo mixes the heads)
```

The four projections (`wq` / `wk` / `wv` / `wo`) are 26 x 26 at any
`n_heads` — same parameters, same FLOPs; more heads = more, narrower
attention patterns per bin pair.

`compile_mode` (optional, flat) sets the CUDA `torch.compile` mode; the
defaults are precedence
[F](#15-appendix-precedence--who-wins-when-settings-collide).

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

The symmetric `trunk:` / `head:` blocks are **diffs** against the top level:
each configures its own pass over the eight keys `lr` / `scheduler` / `loss` /
`trim` / `focus` / `clip` / `rewind` / `ema` (each absent = the run default),
with the per-key override semantics in precedence
[B](#15-appendix-precedence--who-wins-when-settings-collide) — the head block
alone also takes the `activation:` pin alias (`trunk: activation:` is an
error, precedence
[A](#15-appendix-precedence--who-wins-when-settings-collide)). On a
single-phase model (any `resmlp`) `train()` demotes these — `trunk:` merges
into the top level, `head:` / `trunk_epochs` / `freeze_trunk` are dropped —
so the same YAML drives both families ("what is in the trunk is just the
global").

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

## 12. Appendix: the pipeline

The goal is to replace an expensive physics code with a network that maps a
handful of cosmological parameters to the cosmic-shear data vector, fast enough
to call inside a cosmological inference and accurate enough that the data
vector's [**chi2**](#13-appendix-the-chi2-metric-mahalanobis) — its distance from
truth measured in the data covariance (a Mahalanobis distance; see the appendix),
the quantity inference actually cares about — stays small. Two ideas run through the
whole pipeline. **Whitening**: both the inputs and the outputs are rotated and
rescaled so the network sees a decorrelated, unit-variance, well-conditioned
problem instead of raw correlated numbers. **The chi2 metric**: training and
evaluation are judged in the covariance's natural geometry, not in raw
per-element error, because that is what an analysis uses.

```
cosmological parameters
   │   geometries_parameter.py   center, rotate, unit-scale          (whiten in)
   ▼
whitened inputs
   │   emulator_designs.py       ResMLP, ResCNN, or ResTRF
   ▼
whitened data vector
   │   geometries_output.py      un-whiten + scatter to full length  (whiten out)
   ▼
physical residual vs truth
   │   loss_functions.py         contract with the inverse covariance
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

**2. Whiten the inputs** (`geometries_parameter.py`). Raw cosmological parameters
are correlated and span wildly different scales. `ParamGeometry` centers them,
rotates into the parameter-covariance eigenbasis, and scales each direction to
unit variance, so the network receives decorrelated, unit-variance inputs rather
than strongly correlated physical numbers.

```
2.  Whiten the inputs                          ParamGeometry · geometries_parameter.py

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

**3. Whiten the output, and keep the metric** (`geometries_output.py`). The data
vector is *masked* (the analysis keeps only some entries) and strongly
correlated. `DataVectorGeometry` *squeezes* to the unmasked entries and whitens
them in the data-covariance eigenbasis, so every network output is decorrelated
and equally hard to fit. The same object holds `Cinv`, the masked inverse
covariance the chi2 contracts against — geometry and metric live together.

```
3.  Whiten the output, keep the metric         DataVectorGeometry · geometries_output.py

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

**4. Build the loss** (`loss_functions.py`). `make_chi2` wraps the output
geometry in a chi2. Because the targets are whitened, plain squared error in the
whitened space *is* the chi2, so the optimization is well-conditioned; the loss
un-whitens the residual and contracts it with `Cinv` to report the true chi2, and
adds optional robustness (trim the worst points, up-weight the still-hard ones).

```
4.  Build the loss                            make_chi2 · loss_functions.py

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

**5. Choose the model** (`emulator_designs.py`, `IA/emulator_designs.py`).
`ResMLP` is the baseline: an input projection, a stack of residual blocks, an
output projection. `ResCNN` adds a 1D-CNN correction on top of the ResMLP trunk,
acting in *theta order* so a convolution can exploit smoothness along the angular
axis. Two orthogonal YAML keys pick the class: `train_args.model.name` is the
architecture (`resmlp` | `rescnn` | `restrf`), and the separate `train_args.model.ia` key
layers a factored intrinsic-alignment design on it (omit for the plain
emulator; `ia: nla` (1 amplitude, 3 templates) or `ia: tatt` (3 amplitudes,
10 templates) makes the model emit templates the loss combines in closed
form, so the IA amplitudes never enter the network — `TemplateMLP` for
`resmlp`, `TemplateResCNN` for `rescnn` (whose gated conv corrects each
template before the combine), and `TemplateResTRF` for `restrf` (the
bin-token transformer correction head)).

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
train_single  tune_single  sweep_ntrain  sweep_hyperparam  bakeoff_activation
  one run    Optuna search f(dchi2) vs N  one YAML knob      one curve per act
             (multi-GPU,   (multi-GPU,    (multi-GPU,        (multi-GPU, by act)
              journal)      LPT, --gpu-    even split,
                            pack)          --gpu-pack)
```

---

## 13. Appendix: the chi2 metric (Mahalanobis)

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

## 14. Appendix: activation functions

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

## 15. Appendix: precedence — who wins when settings collide

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
| `activation` | head only: an alias for `model.<head>.activation`, consumed at construction (not a training knob) | legal in `head:` (the head trains only in phase 2); an error in `trunk:` (the trunk is the same modules in both phases, so a phase-local trunk activation cannot exist — the error teaches this) |

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

Why: a sweep varies exactly one concretized leaf per point over the shared
baseline; axes that would be silently dropped, or that change the class, are
refused up front.

### F. Constructor / driver args vs the YAML

| setting | precedence |
|---|---|
| `device` | an explicit arg beats auto-detect (CUDA > MPS > CPU) |
| `thresholds` | a constructor arg beats `DEFAULT_THRESHOLDS` |
| `rescale` | a driver flag only (no YAML key) |
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

## 16. AI-Usage

AI Usage: This library (under the `dev` folder) was developed with Claude
Code assistance. However, Prof. Miranda heavily influenced the code at every
level, from macro-designed implementation and changes to minute Python
choices.
