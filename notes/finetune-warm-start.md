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

## Implementer resume (FTW execution, 2026-07-09, Opus)

**Status: coded + Mac-verified to the limit torch allows; diffs uncommitted
on branch claude/amazing-keller-e798b6 (base 7bf9549). The user runs
git.** All ten design rules D-FT1..D-FT10 implemented.

**What landed (4 new files, 6 edited):**
- `emulator/warmstart.py` (new, 735 lines): the finetune subsystem.
  `validate_finetune_config` (D-FT1/D-FT2 loud errors), `resolve_source_root`
  (D-FT1 $ROOTDIR path resolution), `load_source` (wraps
  `rebuild_emulator(compile_model=False)` + reads recipe / rescale attr /
  data block from the h5; enforces plain-ParamGeometry / ia-None / no-pce /
  rescale-none), `recipe_to_model_opts` (architecture inherited; mirrors
  rebuild's factory reconstruction), `extend_input_geometry` (D-FT3 block
  extension, a plain ParamGeometry), `pin_output_geometry` (D-FT4 reuse +
  checks), `transfer_state_dict` (D-FT5 verbatim + zero-padded columns,
  shape-driven, returns padded-key set), `build_warm_start` (D-FT5+D-FT7
  orchestration: builds the eager template via make_model(compile None),
  transfers, runs the parity gate, returns init_state + one verdict line).
- `emulator/training.py`: `make_model` + `run_emulator` gain `init_state=None`
  (D-FT6); loaded strict into the eager module before torch.compile; nothing
  else in the loop changed.
- `emulator/experiment.py`: finetune branches in `from_config` (validate +
  load source once + build on the source class), `build_geometry` (extend
  input + pin output + make_chi2 rescale none, no cosmolike), `build_specs`
  (model_opts from the recipe; resolved_model = source recipe with the new
  input width), `train` (parity gate + init_state pass-through + the resolved
  finetune block folded into resolved_train), and one `print_design` banner
  line. `self._finetune` guards every branch (absent = byte-identical).
- `train_single_...py`: the `attrs["model"]` source moved to `exp.arch` (works
  with no model: block; identical for a plain run) + `finetuned_from` /
  `finetune_extra_names` root attrs on a finetune run only (D-FT8).
- `example_yamls/finetune_emulator_cosmic_shear.yaml` (new) + a pointer block
  in the train_single YAML.
- `gates/checks/finetune_identity.py` (new, 450 lines): the FTW-A check, fully
  synthetic (torch only, no cosmolike), all five acceptance items + the pin
  checks. `gates/board.py`: `gate_ftw_a` + `gate_ftw_b` + two BOARD rows.
  `gates/configs/finetune-smoke-config.yaml` (new) +
  `gates/board_config.json` gate_configs entry.

**Mac gates run (torch is absent on this Mac: no conda / pyenv / venv,
homebrew python has numpy only, so the torch-executed finetune-identity gate
could NOT be run here):**
- `py_compile` / `compileall` on the whole tree: PASS.
- Numpy math exec-probe (mirrors extend_input_geometry + transfer_state_dict
  line-for-line, float32): ALL PASS. The shared-coordinate encoding is
  bit-identical (max|dv| = 0.00e+00) for appended extras (de-risks the gate's
  torch.equal item 1); E orthogonal; n_x=0 byte-identical; transfer padded
  set / verbatim / zero-pad correct; wrong-shape raises.
- AST cross-checks: every `warmstart.*` call site matches the real
  signatures; every cross-module import in warmstart + the gate resolves to a
  real definition; BOARD loads (21 gates, no dup ids, deps satisfiable).
- `CosmolikeChi2.dest_idx` forwards to `geom.dest_idx`, so the template model
  (warmstart) and the training model (run_emulator) get the same output_dim,
  the init_state strict load matches.

**Open items for the Architect / workstation (flagged, not silently
absorbed):**
1. The torch-executed `finetune-identity` gate must run green on a
   torch machine (Mac dev box has no torch). Static + numpy evidence is
   strong; the real torch.equal / parity run is the confirmation.
2. Tier deviation: the spec put `finetune-smoke` in the new-features tier,
   but its source artifact is the emulator `save-rebuild-drift` persists
   (`<driver_root>/chains/gates_emul_evaluate`), which only exists in the
   save-and-sample tier. I registered `finetune-smoke` there with
   `deps=("save-rebuild-drift",)` so its source is guaranteed present, and
   its config `from:` points at that path. If the Architect wants it in
   new-features, a dedicated source-training step is needed first.
3. `finetune-smoke` acceptance: the gate asserts run-completes + the parity
   verdict line + the banner from the log; the `finetuned_from` root attr and
   the save->rebuild->predict round-trip are logged as the workstation leg
   (the gate cannot read the dynamic saved-artifact path from stdout).

## Architect audit verdict (FTW, 2026-07-09, Fable)

**VERDICT: VERIFIED — sign-off given.** Audited at main e4cf27a (the user
fast-forwarded the branch before the audit; findings below are rulings and
pre-authorizations, none blocking). Evidence, all mine against the shipped
source, never the Implementer's summaries:

- **Independent exec-probe of the shipped spans** (ast-extracted
  `extend_input_geometry` / `transfer_state_dict` from main e4cf27a, run
  under numpy shims on the Mac): 15/15 PASS, including two layouts the
  check script does not cover — INTERLEAVED extras
  (`names_n = [w0, a, b, wa, c]`: name-keyed row placement, E orthogonal to
  3.9e-08, shared coords bitwise-equal to the source encoding on this BLAS,
  cross-independence bitwise both ways) and the PERMUTED-order degenerate
  case (encoding equals the source to 2.4e-07 — correct, and instructive:
  NOT bitwise, see the pre-authorization below). Transfer: padded set exact
  on 2-D and 3-D input consumers, verbatim tensors bitwise, n_extra=0
  verbatim, all three raise paths fire.
- **Wiring read line-by-line in the diff:** init_state loads strict into
  the eager module BEFORE torch.compile (make_model); optimizer / scheduler
  / EMA / trim / focus untouched (fresh); the finetune branches guard on
  `self._finetune` (absent = byte-identical plain run); the parity gate runs
  on the same state dict the training model loads. The six train_args
  sub-blocks the finetune build_specs branch spreads are a PRE-EXISTING
  schema requirement (the normal path calls build_run_specs identically) —
  no new demand on YAMLs.
- **The smoke config's source path is the board-proven convention:**
  `projects/lsst_y1/chains/gates_emul_evaluate` is byte-identical to the
  emulators entry in cobaya-adapter-evaluate.yaml, which read the same
  artifact on the green run-10/11 board; gsv_bitwise_drift.py persists it
  with no run-tag suffix. All ctx helpers the new gates call (run_check /
  run_driver / require_config / logscan.search) exist with those signatures
  and are used identically by existing gates.

**Rulings on the declared deviations:**
1. **Tier move APPROVED.** `finetune-smoke` lives in the save-and-sample
   tier with `deps=("save-rebuild-drift",)` — dependency-correct, since its
   source artifact only exists after that gate persists it. This supersedes
   the spec's "new-features tier" phrase above; no dedicated source-training
   step is wanted.
2. **Log-visible acceptance APPROVED.** The `finetuned_from` attr and the
   save->rebuild round-trip stay a one-time Architect confirmation from the
   workstation artifact, not a check script: save-rebuild-drift already
   proves the round-trip machinery in general, so a dedicated script would
   re-prove it for low value. The leg: after the board runs, take the
   "saved run record -> <path>.h5" line from the finetune-smoke log and run
   `python -c "import h5py,sys; f=h5py.File(sys.argv[1]);
   print(f.attrs['finetuned_from'], repr(f.attrs['finetune_extra_names']))"
   <path>.h5`.

**Pre-authorization (recorded so a board red is triaged in one step):** the
check item `encode: shared coords bit-identical to the source encoding`
(finetune_identity.py, torch.equal across DIFFERENT matmul widths, n_s vs
n_n) is a backend-dependent assertion: a kernel that regroups the reduction
across k can break bit-equality while the math is exact (my permuted-order
probe showed exactly this at 2.4e-07). If finetune-identity ever fails on
that label ALONE with max|delta| <= 1e-6, relax that single assertion to a
1e-6 tolerance and note it here — no re-audit needed. Every OTHER bitwise
assertion (extras-independence, verbatim transfer, zero columns, same-order
degenerate) compares same-shape outputs of the same kernel and must hold
bit-for-bit on any backend.

**Non-blocking nit, recorded:** the train() comment beside the parity
verdict says it prints "even under a quiet run", but `self.log` is
quiet-gated — the behavior (prints whenever the experiment is not quiet,
which the board runs satisfy) is fine; the comment overstates. Fix whenever
that file is next touched; not worth a commit alone.

**Remaining before FTW closes:** (1) the user runs the board on the
workstation — the 19 green gates skip, the two new FTW gates run; relay the
raw logs; (2) the one-time finetuned_from/attr confirmation above; (3) the
n_x > 0 leg on real data waits for a real w0waCDM dump (the honest margin
already recorded in the gates section).

## Board run 12 (2026-07-10) + delta D-FTW-1 (Architect, applied directly)

**finetune-identity (FTW-A): PASS on CUDA, 15/15 — CLOSED.** The raw log
shows every assertion green, including the two the audit flagged as
backend-dependent: the shared-coordinate encoding held BITWISE on CUDA
(max|dv| = 0.00e+00) and the parity was exactly zero on 256 rows. The
pre-authorized 1e-6 relaxation was never needed; it stays on file for other
backends.

**finetune-smoke (FTW-B): FAIL at load_source, one step before the design
ran** — `rescale=None`: the board's source artifact
(`<driver_root>/chains/gates_emul_evaluate`, persisted by the
save-rebuild-drift CHECK SCRIPT, not the training driver) carries no
`rescale` root attr, and the D-FT2 gate rejects it. The synthetic FTW-A
source stamps `attrs={"rescale": "none"}` explicitly, which is why the
identity gate never exposed this. **The gap is the Architect's (spec-level
integration): D-FT2 demanded the attr without verifying the board artifact
carries it.** The Implementer implemented the spec faithfully.

**Delta D-FTW-1, applied directly by the Architect (declared deviation, two
fully-determined edits, compile-checked):**
1. `gates/checks/gsv_bitwise_drift.py` — the persistent save now stamps
   `"rescale": exp.rescale` (the RESOLVED run value, never a literal) in
   its attrs, beside n_train.
2. `emulator/warmstart.py` load_source — the missing-attr case is now its
   own loud error (distinct from a wrong value): names the h5, explains the
   check-script provenance, and points at `--force-rerun save-rebuild-drift`.
   Still NO fallback to "none" — an artifact that does not record its
   rescale is ambiguous (never-trust-defaults).

**Forward-walk of the remaining smoke path (audited before the fix, so the
rerun is expected green in one pass):** the GSV artifact's config carries
the matching cosmolike keys and the SAME dumps/covmat as the smoke config
(names equal -> the n_x = 0 degenerate warm start), the recipe rebuilds
gated_power/n_gates 3 with compile_mode None inherited, widths 12 / 1560
match the pin checks, and parity is bit-trivial. Rerun:
`python gates/run_board.py --force-rerun save-rebuild-drift` (one pass:
save-rebuild-drift forced -> re-persists the artifact with the attr;
finetune-smoke reruns by default since it is recorded FAIL; everything else
skips green).

**Run-12b amendment:** the D-FTW-1 rerun tripped a SECOND latent gap, this
one GBC-side (the GSV check script self-reads the raw board_config.json and
saw the shipped rootdir null) — fixed as delta D-GBC-1, recorded in
gates-harness-user-run.md. The FTW forward-walk above is unaffected; the
smoke path itself has not executed yet.

## Run 12c (2026-07-10) + delta D-FTW-2 (Architect, applied directly)

The D-GBC-1 rerun got two steps further — save-rebuild-drift PASS (the
artifact now persists WITH the rescale attr; all bitwise legs green), the
smoke's load_source accepted it and the banner printed — then died in
print_design at `ta['model']`: the finetune branch added the banner line
but the function's model-spec line still assumes the model: block a
finetune YAML is FORBIDDEN to carry. **Both the Implementer's static sweep
and the Architect audit missed it** (the audit read the print_design diff
hunk, not the whole function body). This time the ENTIRE driver path was
enumerated (every `model`-block read in experiment.py + the driver):
exactly two live crash sites existed — print_design:1334 (fired) and the
driver's run_tag:199 (lying in wait at save time, unavoidable since --save
defaults to "emulator").

**Delta D-FTW-2 (compile-checked + exec-probed):**
1. `experiment.py` print_design — on a finetune run the model-spec line
   prints the inherited source recipe's constructor kwargs (the consumed
   view of D-FT2) instead of describe_spec on the absent block.
2. `train_single_...py` run_tag — the architecture tag reads `exp.arch`
   (the resolved value; the same move the Implementer already made for the
   save attrs). Probe on the shipped span: the plain-run tag is
   byte-identical to the old output; the finetune-run tag works
   (resmlp_t16_ntrain200).

All other model-block reads are on paths the finetune branch returns
before, or safe `.get(..., {})` — enumerated, not assumed. Deterministic
consequence for the workstation leg: the smoke artifact lands at
`<rootdir>/projects/lsst_y1/chains/emulator_resmlp_t16_ntrain200.h5`.
Rerun: plain `python gates/run_board.py` (no --force-rerun needed:
save-rebuild-drift is recorded PASS, finetune-smoke reruns from FAIL).

## FTW closure (2026-07-10): board green, both gates PASS

**finetune-smoke (FTW-B): PASS** (run 12d, HEAD ebfb16c). The raw log is
the design executing end to end:

- `[ok] finetune parity: max|dv| = 0.000e+00 on 200 rows` — epoch 0 IS the
  source function, bitwise, on the names-equal path; the "200 rows" (not
  256) shows the min(available-rows) guard doing its job on the tiny set.
- banner + inherited-recipe spec line printed (the D-FTW-2 branch:
  `model spec: inherited from the source recipe {int_dim_res 64, ...}`).
- lr 7.07e-05 = 1e-4 * sqrt(32/64): the lowered lr_base through the
  existing sqrt-batch rule, no new knob (D-FT1 as designed).
- val decreased gently from the warm-start baseline (23210.9 -> 23203.5 ->
  23196.1 over 2 epochs) — fine-tuning moves OFF the source point with no
  cold-optimizer kick (warmup 1 epoch, fresh Adam moments).
- artifact saved at the predicted path:
  `<rootdir>/projects/lsst_y1/chains/emulator_resmlp_t16_ntrain200.{emul,h5}`.

The full delta trail this green retroactively proves: D-FTW-1 (rescale
attr stamped + read), D-GBC-1 (check-script rootdir resolution), D-FTW-2
(print_design + run_tag off the forbidden model block). **Board: 21/21
green** (triangle-shading remains the standing optional eyeball).

**One item left before FTW-B is formally closed** — the deviation-(b)
workstation leg, one command on the workstation:

    python -c "import h5py; f = h5py.File('$ROOTDIR/projects/lsst_y1/chains/emulator_resmlp_t16_ntrain200.h5'); print(repr(f.attrs['finetuned_from'])); print(repr(f.attrs['finetune_extra_names'])); print(repr(f.attrs['rescale']))"

Expected: finetuned_from = the absolute gates_emul_evaluate root,
finetune_extra_names = '' (names-equal run), rescale = 'none'. The
save->rebuild round-trip machinery is already proven bitwise on this
artifact class by save-rebuild-drift in the same board pass.

After that: FTW is done for V1. The n_x > 0 leg on real data (LCDM ->
w0waCDM proper) opens with the science thread, when a real w0waCDM
training dump exists — the honest margin stands.

**Attr confirmation (2026-07-10, user-run on the workstation):**
finetuned_from = '/home/vivianmiranda/.../projects/lsst_y1/chains/
gates_emul_evaluate' (the absolute source root), finetune_extra_names = ''
(names-equal run), rescale = 'none'. All three exactly as specified —
**FTW V1 CLOSED.** Both gates green on the board, provenance persisted and
verified from the artifact itself. The unit's one open thread is the
n_x > 0 real-data leg, which belongs to the science thread (needs a real
w0waCDM training dump).

## Designed extension (2026-07-10): finetune.anchor — L2-SP for warm starts

User-requested after the TPE D-TP10 discussion: the same anchor penalty
lambda * ||W - W_source||^2 applies to plain fine-tuning, where it is the
missing continuous dial between frozen and free (the lowered LR sets the
step size, not how far the walk may drift from the proven source).

- YAML: `finetune.anchor` (optional; the whitelist grows by one key).
  ABSENT = the shipped, gated behavior, byte-identical. Present =
  explicit lambda (0.0 states free fine-tuning deliberately).
- The reference is the transferred init_state, with the PADDED EXTRA
  COLUMNS EXCLUDED from the penalty: they are exact zeros by design and
  the designated carriers of the new-physics dependence — anchoring them
  to zero fights the warm start's purpose. transfer_state_dict's
  padded_keys already identifies them; the mask is a column slice.
- Mechanism, SHARED with TPE D-TP10 (one facility, built once): a
  DECOUPLED post-step update W <- W - lr * lambda * (W - W_0), the
  AdamW-decoupling argument applied to the anchor (a loss-term L2 gets
  rescaled by Adam's adaptive moments). Interaction documented: the
  existing weight_decay decays toward ZERO, i.e. away from the source —
  an anchored fine-tune should normally set weight_decay 0.0 (a
  recommendation with a notice, never an error; the user decides).
- Executes WITH TPE unit 2 (the shared anchor facility lands once,
  both YAML surfaces gain their key in the same unit); FTW stays closed
  until then, and absent-key behavior never changes.
