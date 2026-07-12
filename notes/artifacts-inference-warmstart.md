# Artifacts, inference, adapters, and warm starts

Consolidated 2026-07-11 from save-schema-resolved-config.md,
cobaya-theory-adapter.md, finetune-warm-start.md,
transfer-parallel-emulator.md, geometry-family-folder.md (retired;
full texts + delta ledgers in git history). Code homes:
emulator/results.py, emulator/inference.py, emulator/warmstart.py,
emulator/losses/transfer.py, emulator/geometries/, cobaya_theory/.

## The standing rule (user verbatim, binding on every save/load surface)

"The philosophy over the emul and h5 file has to be — dont trust on
default values — they can drift." Two halves: WRITE side — everything
the run consumed is written with defaults MATERIALIZED at save time;
READ side — reconstruction reads ONLY the file, a missing key is a
loud error naming it, NEVER a code-default fallback. Third leg of the
consumed-view doctrine: displays RENDER, artifacts PERSIST, loaders
TRUST ONLY it.

## Schema v2 (the live artifact contract)

- One emulator = one path root -> `<root>.emul` (cpu state_dict,
  _orig_mod stripped) + `<root>.h5`.
- The h5 holds: raw config_yaml + train_args_yaml (provenance of what
  was WRITTEN); `config_resolved_yaml` (resolved_train + resolved data
  block); a `model_recipe/` group (class qualname, every constructor
  kwarg actually passed, callables serialized by name, constructor
  defaults materialized via inspect.signature); the geometry state
  groups (param_geometry, dv_geometry, + pce / transfer_base when
  present); histories; root attrs schema_version=2, git commit, torch
  version, rescale, family facts.
- EVERY geometry group carries a `"cls"` attr = full module path
  (D-CT1): rebuild importlib-resolves the stored string and calls THAT
  class's from_state; a missing marker is a loud KeyError naming a
  re-save, never a base-class fallback. The file records WHAT it is,
  not just its numbers.
- `rebuild_emulator(path_root, device)` (results.py): h5-only; v1
  files refused loudly; returns (model, pgeom, geom, info) with info
  carrying ia / pce / transfer / family facts.
- Head artifacts rebuild from files alone (2026-07-11): the family
  geometries re-derive their split (attach_head_coords, inside
  _rebuild_model); the cosmolike DataVectorGeometry PERSISTS it —
  state() writes bin_sizes (+ pm_kept) when build_shear_angle_map
  attached them (schema-additive, the section_sizes/probe pattern;
  __init__ kwargs attribute-UNSET when None so the hasattr guards
  survive). A pre-persistence head file is refused loudly
  ("bin-split persistence"); rebuild never re-derives the cosmolike
  split — that would need ROOTDIR data files at inference.
- Acceptance currency: save -> rebuild -> BITWISE-equal prediction,
  plus the DRIFT TEST — monkeypatch a sharp code default
  (make_activation n_gates 3->7) and rebuild unchanged. The GSV-A
  census mechanically diffs every run_emulator knob and model kwarg
  against the recipes, so a future knob that skips the recipe fails
  the gate. D-SV1/D-SV2 were both "latent drift channels inside the
  anti-drift unit" (a hardcoded compile_mode duplicate; an eval_bs key
  nothing wrote).
- GEO (2026-07-11): the geometry classes live in
  emulator/geometries/{parameter,output,scalar,cmb,grid,grid2d}.py.
  The move shipped with flat shims for the old paths; D-GEO5 (user
  ruling: only test artifacts existed) DELETED the shims — the old
  flat paths now die loudly (ModuleNotFoundError naming the path), the
  geo-paths gate pins new-save markers + dead old paths + a tree
  census. Fresh saves write folder paths via type().__module__.

## Inference: EmulatorPredictor + the five cobaya adapters

- Three layers: EmulatorPredictor (inference.py) owns ALL prediction
  physics; each cobaya_theory/ adapter is THIN (no nn.Module, no
  physics); the MCMC YAML names path roots and nothing else. Retired
  legacy conventions: ord (geometry names ARE the requirements — the
  authority chain has no second list), extrapar (model_recipe),
  duplicated architectures, manual whitening.
- Adapters: emul_cosmic_shear (dv; `dv_return: section|3x2pt`, default
  section — the likelihood glues per-probe sections; section_sizes +
  probe persisted in the geometry), emul_scalars (derived params),
  emul_cmb (get_Cl), emul_baosn (Hubble + distances, piecewise by
  window), emul_mps (Pk grid/interpolator, EMUL2). All five mutually
  reject wrong artifact kinds NAMING the right adapter.
- THE python_path TRAP: cobaya loads an external Theory class via
  `python_path`, NOT `path` — without it a LEGACY v1 adapter bundled
  in cocoa's cobaya fork silently shadows the class.
- The predictor's decoder branches per family and reuses the EXACT
  training decode (factored ia -> TemplateFactoredChi2.decode; NPCE
  residual/ratio; cmb -> the amplitude-law decoder; grid -> {"z",
  quantity}; grid2d -> LAW-SPACE {"z","k",surface} — the syren base
  multiply-back is emul_mps's job). Transfer branch goes FIRST (wins
  over the ia branch a factored correction would otherwise take).
- rescale runs (analytic-R) are OUT of predictor scope: R needs
  cosmolike at inference — a documented h5-only limitation.
- MPS-device caveat: geometry whitening tensors are float64-heritage;
  Apple MPS has no float64 — inference there may need a downcast;
  cuda/cpu are the documented targets.
- Direct scripting (no cobaya): README appendix "scripting a saved
  emulator" — EmulatorPredictor two-door pattern; the background
  family pairs with emulator.background.distance_interpolators.

## Fine-tuning (FTW; universal across families)

- `train_args.finetune: {from, compile_mode?}`; architecture inherited
  from the source h5 (a model: block beside finetune: is a loud
  error); lower LR through the ordinary lr: block (recommend one
  decade down + warmup_epochs >= 3 — fresh cold Adam moments).
- THE invariant: at epoch 0 the warm-started model computes EXACTLY
  the source function, independent of the new parameters' values —
  checked by the parity gate (max|dv| <= 1e-5 float32; 0.000e+00 on
  names-equal runs).
- The mechanism (warmstart.py): block-extended input geometry — source
  rotation verbatim on shared rows, extras whitened by their MARGINAL
  covmat block, encoded layout [shared ; extras ; raw amps]; the
  shared coords are BIT-identical to the source encoding. State
  transfer is shape-driven: equal shapes copy verbatim; dim-1 grows by
  exactly n_x -> source columns + EXACT-ZERO new columns. Output
  geometry PINNED from the source artifact (class-preserving via the
  cls marker). Accepted tradeoff: extras-shared cross-correlations are
  NOT whitened away (full decorrelation would destroy exactness).
- Family branches: every family fine-tunes (scalar/cmb/grid/grid2d
  pin the SOURCE output geometry wholesale after compatibility
  checks; wrong-kind + metadata mismatches loud).
- `finetune.anchor` (optional L2-SP): decoupled post-step
  W <- W - lr*lam*mask*(W - W_0) with the padded extra columns
  EXCLUDED from the penalty (they carry the new physics); never a loss
  term (Adam's moments would rescale it); weight_decay-0 recommended.
- Provenance attrs: finetuned_from + finetune_extra_names.

## Transfer learning (TPE; family-wide since 2026-07-12, scalar excepted)

- Scope (RE-RULED 2026-07-12, overnight): the user overturned the
  BAOSN/MPS permanent forbid and D-CM7's deferral — "I misspoke -
  this for sure should be allowed for MPS. And it is easy to allow it
  to BAO/SN - because it is weird to have a feature not symmetric to
  all cases." Transfer now rides cosmolike + cmb + grid + grid2d.
  The one family still out is SCALAR (D-SP8 stands — a recorded
  ruling, not a structural bar; overturning it is the user's call).
- Concept: the trained base is FROZEN WHOLE; a small parallel
  correction net sees the FULL new parameter space; composition
  `gain` = base*(1+r) or `sum` = base + r, in `space` physical or
  whitened (absent space resolves to the form's recommendation and is
  MATERIALIZED). The model: block describes the CORRECTION net.
- The diagonal families use losses/transfer.py::TransferDiagChi2
  (subclasses CmbDiagonalChi2): plain bases only, space WHITENED only
  (their metric basis; explicit physical is loud — an elementwise
  scale away, or a log-law domain edge), both forms with a gain
  zero-crossing notice (sum recommended), transfer.refine rejected
  (frozen-base V1), roughness+transfer refused loudly, and on cmb
  only amplitude_law "none" both sides (one target construction at a
  time). Base pins mirror the finetune pins (spectrum/ell/sigma; z;
  z+k; + quantity/units/law equality); a cross-family base is a loud
  from_config error. build_transfer_start (the D-TP7 parity gate)
  rides unchanged — it duck-types on decode/base_decode. Legs:
  check_diagonal in transfer-identity.
- Why not FTW for new physics sectors: same-capacity adaptation is
  structurally insufficient; the metric is SAMPLE EFFICIENCY
  (accuracy per training cosmology — extended-model dumps are the
  expensive object), never wall-clock.
- Speed design: the frozen base runs ONCE per row at encode, packed
  [base ; truth] into the staged target; the hot chi2 composes, never
  re-runs the base (hook-counted).
- Identity invariant: correction ≡ 0 -> composed prediction ==
  frozen base decode BITWISE — except the factored-PHYSICAL leg,
  where combine/unwhiten reassociation gives ~4e-6: bitwise is
  demanded only on same-computation legs; cross-path legs relax to a
  documented ~1e-6/1e-5 (ruled three times).
- Artifact: the base is EMBEDDED whole (transfer_base group: recipe +
  state + both geometries + form/space), never referenced; chaining
  refused. `transfer.refine` (optional stage 2, ULMFiT-style):
  unfreeze once, per-group LR (base_lr_scale), REQUIRED explicit
  anchor lambda (0.0 must be stated); refined artifacts keep the
  PRETRAINED W_0 in transfer_base + the drifted base in
  drifted_state (two-way consistency loud; drift norms recomputable
  from the file's two states; predictor picks drifted silently).
- Four training modes, one dial: from-scratch, anchored warm-start,
  frozen-base transfer, anchored joint refinement — the decoupled
  L2-SP lambda spans frozen to free.

## Follow-the-IDs (git archaeology)

FTW: D-FT1..10, D-FTW-1/2. TPE: D-TP1..10, D-TPE-1, D-TPE2-1..3,
Ruling 1 (reassociation). Schema: D-SV1/2, D-CT1..3, GCT-D
(dv_return), Riders 1-5 (paths, .paramnames cross-check, GitHub math
policy). GEO: D-GEO1..5. Board homes here: save-rebuild-drift,
cobaya-adapter, finetune-identity/smoke, transfer-identity/smoke,
geo-paths.
