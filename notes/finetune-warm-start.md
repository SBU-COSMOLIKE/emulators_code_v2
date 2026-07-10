# Fine-tune warm start across cosmologies (spec)

**Date:** 2026-07-09. **Status:** SPEC (Architect, Fable) — handed to the
Implementer. **Spec code:** FTW. **Home note** for the gates
`finetune-identity` / `finetune-smoke`.

## The request (user design goal)

Training always starts from random weights. Add the capability of loading a
saved emulator (say, one trained on LCDM) and fine-tuning it — continuing
training with a lower initial learning rate — on a new training set (say,
w0waCDM). The classic use: the new parameter space contains the old one plus
extra parameters (w0, wa).

## The core problem, and the design principle

Loading the weights is one line. The real problem is that the artifact's
whitening bases are cosmology-dependent:

- The input encoding is a full eigenbasis rotation
  (`ParamGeometry.whiten`, geometries_parameter.py):
  `encode(theta) = ((theta - center) @ evecs) / sqrt_ev`, where evecs /
  sqrt_ev come from eigendecomposing the training covmat. A w0waCDM covmat
  does not just add two columns — its eigendecomposition remixes ALL
  parameters, so under a naively rebuilt geometry the loaded input weights
  point at scrambled coordinates and the run starts far from the source
  function, mostly relearning the basis.
- The output whitening basis and Cinv come from the DATASET covariance +
  mask (`DataVectorGeometry.from_cosmolike`) — cosmology-independent. Only
  `center` (the training-mean dv) differs between dumps.

**Design principle (the invariant everything below serves):** at epoch 0 the
warm-started model must compute EXACTLY the source emulator's function,
independent of the new parameters' values. This is the same loss-continuity
trick the two-phase schedule already uses (the zero-init head that starts as
an exact identity). Fine-tuning then moves away from a proven starting point
instead of a scrambled one, and the pre-train parity gate (D-FT7) makes the
exactness a checked fact, not a hope.

## Design rules

### D-FT1 — YAML surface: one new block, no duplicated knobs

```yaml
train_args:
  finetune:
    from: projects/lsst_y1/emulators/lcdm_run/emul_v2
  lr:
    lr_base: 5.0e-4
    bs_base: 64.0
    warmup_epochs: 5
```

- `finetune.from` (required, string) = the source artifact path root
  (`<root>.h5` + `<root>.emul`, as written by save_emulator). Resolution:
  expanduser/expandvars; a relative path joins `$ROOTDIR` (the cobaya
  adapter's convention); relative with `$ROOTDIR` unset is a loud error.
  Either file missing is a loud error naming the full resolved path.
- `finetune.compile_mode` (optional) = torch.compile mode override for this
  machine (`reduce-overhead` / `default` / null); absent -> the mode
  persisted in the source recipe. This is the ONE architecture-adjacent key
  allowed here, because compile mode is a machine knob, not an architecture
  knob (it mirrors rebuild_emulator's compile_model parameter).
- Any other key under `finetune:` is a loud error listing the whitelist.
- **The lower learning rate is NOT a new key.** The existing `lr:` block
  already owns it; a fine-tune YAML simply writes a smaller `lr_base`
  (one decade down from the source run is the starting recommendation) and
  keeps `warmup_epochs >= 3`: the optimizer restarts with cold Adam moment
  estimates, and the warmup is what protects the loaded weights from the
  first-steps kick.

### D-FT2 — the architecture is inherited, never restated

A finetune run reads the model recipe from the source h5 (`model_recipe`,
the same record rebuild_emulator consumes; a missing key is loud, never a
code default). Consequences, all loud errors at config time:

- a `model:` block beside `finetune:` — the error says the architecture
  comes from the artifact and shows the key to delete;
- an explicit `--activation` driver flag;
- a `pce:` block (NPCE composition is out of scope, D-FT10);
- `--rescale` other than `"none"` — and the SOURCE h5 root attr `rescale`
  must also be `"none"`;
- `trunk_epochs > 0`, `trunk:`, `head:`, or `freeze_trunk` — a warm start
  is already past the trunk-warming era; V1 is single-phase only;
- source artifact constraints: schema v2 (rebuild_emulator already
  refuses v1), `param_geometry` cls exactly `ParamGeometry` (not Log, not
  AmplitudeFactor — V1), `info["ia"]` None, no `pce` group.

The source is loaded ONCE per experiment via `rebuild_emulator(...,
compile_model=False)` and cached: it supplies the recipe, the source
state_dict, both source geometries, and the parity-reference model.

### D-FT3 — input geometry: block-extension, exact on shared parameters

Let the source artifact store `names_s` (n_s params), `center_s`,
`evecs_s = V` (n_s x n_s, orthonormal columns), `sqrt_ev_s = s`. Let the
new run's covmat header be `names_n` (n_n params). Require
`set(names_s) <= set(names_n)`; the extras (`names_n - names_s`, n_x of
them, kept in `names_n` order) are the new physics parameters.

Build the run's ParamGeometry as a PLAIN ParamGeometry whose tensors are
block-structured data (no new class, no new save/load code):

```
r(i)   = row of source name i in names_n        (name lookup)
x_j    = row of extra j in names_n

center_e[r(i)]  = center_s[i]                   (source values, verbatim)
center_e[x_j]   = staged-train mean of column x_j (train_set C_mean)

E = zeros(n_n, n_n)
E[r(i), j]      = V[i, j]        for j < n_s    (source rotation, verbatim)
E[x_j, n_s + k] = W[j, k]                       (extras' own whitening)

sqrt_ev_e = concat(s, sqrt(lam_x))

  where  Sigma_xx = the extras' marginal block of the NEW covmat,
         lam_x, W = eigh(Sigma_xx)              (n_x x n_x)

(legend: n_s / n_n / n_x = source / new / extra parameter counts;
 r(.), x_. = row indices in the new covmat header order; E = the extended
 evecs; W, lam_x = eigenvectors / eigenvalues of the extras' marginal
 covariance; center_e / sqrt_ev_e = the extended center / scales.)
```

Invariants the construction guarantees (and the gates assert):

- encoded coordinates `0..n_s-1` of the new geometry are **bit-identical**
  to the source encoding of the shared parameters (the extra rows carry
  exact-zero entries in those columns; `0.0 * anything` contributes exact
  zeros to the sums);
- encoded coordinates `n_s..` depend only on the extras;
- E is orthogonal (two orthonormal blocks with disjoint row support), so
  `unwhiten`/`decode` stay exact inverses;
- degenerate case `n_x = 0` (same names, any order; e.g. LCDM -> LCDM on
  more data): the construction reduces to the source geometry with rows
  keyed by name — same-order input makes it byte-identical to the source's
  tensors. ONE code path covers both fine-tune flavors.

Accepted tradeoff, recorded: cross-correlations between the extras and the
shared parameters are NOT whitened away (we keep only the marginal
`Sigma_xx`). Full decorrelation would require the new eigenbasis, which
destroys exactness. A pre-trained trunk can afford the slightly worse input
conditioning; exactness of the start cannot be traded.

### D-FT4 — output geometry: pinned from the source artifact

The run does NOT call `DataVectorGeometry.from_cosmolike`. The source
dv geometry is reused wholesale (class-preserving via the h5 `cls` marker)
and persisted again in the new artifact — self-consistent under the
persist-resolved-values rule. The pinned `center` remains the SOURCE
training-mean dv: a zero-point convention, harmless for a nested model
family, and it keeps the whole design at exactly one surgery site (the
input columns). (The rejected alternative — recompute center from the new
dump and fold the constant offset into the output bias — is also exact but
adds a second surgery site for no accuracy gain.)

Validation before pinning (all loud errors, no cosmolike needed):

- the source h5 `config_resolved_yaml` data block's `cosmolike_data_dir`
  and `cosmolike_dataset` must equal the new run's data block values
  (same dataset + mask + scale cut is a hard prerequisite of fine-tuning);
- the source dv geometry's persisted `probe` must equal the run's probe;
- the new dump's dv width must equal the pinned geometry's `total_size`.

`make_chi2(rescale="none")` wraps the pinned geometry in a plain
CosmolikeChi2 with no cosmolike calls; restrf's `build_shear_angle_map`
(needs_bins) still runs on the pinned geometry exactly as build_geometry
does today.

### D-FT5 — state-dict transfer: verbatim + zero-padded input columns

Generic, shape-driven, no per-design enumeration. Build the fresh model
(new input_dim = n_n) and use its state_dict as the shape template; for
every key:

- template shape == source shape -> copy the source tensor verbatim;
- tensor ndim >= 2 and the shapes differ ONLY in dim 1 by exactly +n_x ->
  source columns first, then **exact-zero columns** for the extras (this
  catches the input Linear of every design AND the FiLM generators of
  ResCNN / ResTRF, whose first layer is also sized by input_dim);
- anything else -> loud error naming the key and both shapes;
- a template key missing from the source, or a leftover source key ->
  loud error (same architecture guarantees neither happens).

Zero columns + the block-extended encoding are the exactness proof: the
extras' encoded coordinates are the ONLY ones that see them, and they meet
zero weights, so epoch 0 IS the source function, bit-for-bit in eager
arithmetic. With n_x = 0 the transfer is a verbatim strict load.

### D-FT6 — training-loop wiring: load before compile, everything else fresh

- `make_model` gains `init_state=None`: when given, the state dict loads
  strict into the EAGER module before torch.compile (an OptimizedModule's
  prefixed keys never enter the picture).
- `run_emulator` gains `init_state=None`, forwarded to make_model. No other
  training-loop change: optimizer, scheduler, warmup, EMA, trim/focus
  schedules all start fresh (fine-tune is NOT resume), and the loop's
  existing incoming-weights snapshot automatically makes the loaded weights
  the epoch-0 best baseline for selection and rewind.
- The extended geometry's `encoded_dim`/width reaches make_model through
  the existing input-width logic; nothing re-derives n_s anywhere.

### D-FT7 — the pre-train parity gate (one verdict line)

Before training, on the same padded state dict the training model will
load: rebuild an eager module from the source recipe, load the padded
state dict, and against the cached source model assert on >= 256 staged
training rows:

- `max |new(encode_e(theta)) - src(encode_s(theta_shared))| <= 1e-5`
  (float32, whitened-dv units; the widths differ, 12- vs 14-wide matmuls,
  so bit-equality is not demanded here);
- extras-independence: two inputs identical except in the extras produce
  **bit-identical** outputs (`torch.equal`) on the eager path.

Pass prints exactly one line (`[ok] finetune parity: max|dv| = ... on 256
rows`, essential-only terminal rule); failure is a loud error printing the
max deviation. The check then discards its module; the training model loads
the identical state dict strict, so the checked function is the trained one.

### D-FT8 — provenance, persisted and printed

- h5 root attrs gain `finetuned_from` (the resolved absolute source path
  root) and `finetune_extra_names` (space-joined, `""` when n_x = 0).
- the resolved/consumed config records the finetune block with the path
  and compile_mode materialized (persist-resolved-values).
- `print_design` gains one banner line naming the source and the extras.
- The artifact stays standard schema v2: the cobaya adapter needs NO
  change — the h5's stored names now include the extras, and the existing
  loud dependency-resolution error already tells a sampling YAML that
  forgets to sample them.

### D-FT9 — code placement

New module `emulator/warmstart.py` (house style per
[[py-module-style-conventions]]): the finetune-block validator, the source
loader/validator (wrapping rebuild_emulator), the geometry extension, the
state-dict transfer, and the parity check. experiment.py calls into it from
config validation / build_geometry / train; training.py's only change is
D-FT6. Keeps experiment.py (1933 lines) from growing another subsystem.

### D-FT10 — out of scope (V2 candidates, recorded so nobody re-derives)

- factored-IA sources (AmplitudeFactorGeometry) and LogParamGeometry;
- NPCE composition (frozen PCE base is fit to a training set — warm-starting
  the refiner across bases needs its own design);
- freeze schedules (linear-probe phases: freeze all but the input
  projection for k epochs) — the phase machinery exists if ever wanted;
- cross-dataset / cross-mask transfer (different scale cuts change the
  output dimension; a different unit entirely);
- the output-center rebase alternative (rejected above, D-FT4).

## Validation gates

**`finetune-identity` (FTW-A; Mac + board; torch, no cosmolike).** A check
script (`gates/checks/finetune_identity.py`, plain-language docstrings per
[[gates-checks-docs-plain-language]]) that builds a tiny synthetic source
artifact (small ResMLP, synthetic ParamGeometry from a random SPD covmat,
synthetic DataVectorGeometry built directly, save_emulator round-trip),
then runs the warm-start path with two extra names and asserts:

1. shared-coordinate encoding: `torch.equal` against the source encoding;
2. transfer: every non-padded tensor `torch.equal` to the source; padded
   tensors are exactly the source columns + exact-zero extras; the set of
   padded keys is exactly the input-consumer set;
3. parity: `max |delta| <= 1e-5` on 256 rows; extras-independence
   `torch.equal`;
4. degenerate n_x = 0: transferred state dict bitwise-identical to the
   source's; geometry tensors bitwise-identical;
5. error paths raise with the required naming: non-superset names (both
   lists printed), `model:` beside `finetune:`, rescale mismatch.

**`finetune-smoke` (FTW-B; workstation board, new-features tier).** A
names-equal fine-tune from the board's own trained artifact (the training
gate's output under the `gates_board` fileroot): nepochs 2, `lr_base` one
decade down. Acceptance: the parity verdict line passes; the run completes;
the saved artifact carries `finetuned_from`; a rebuild_emulator round-trip
on the fine-tuned artifact predicts identically to the in-memory model (the
save-rebuild-drift pattern). **Honest margin, recorded now:** the extension
path (n_x > 0) is exercised on real data only when a real w0waCDM dump
exists — the Mac gate owns the extension math until the science thread
provides one.

Both gates register in the board (gates/board.py + run_board.py, home note
= this file) with raw logs under gates/logs/.

## Links

[[py-module-style-conventions]], [[gates-harness-user-run]],
[[gates-checks-docs-plain-language]], [[designs-losses-family-folders]]
(the module layout warmstart.py sits beside), [[gates-id-translation]]
(FTW-A/B are the notes-side keys; the code uses the human gate ids).
