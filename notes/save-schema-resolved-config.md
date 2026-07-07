---
name: save-schema-resolved-config
description: "SPEC 2026-07-07 (Architect): save schema v2 — the .emul + .h5 pair must reconstruct the emulator PERFECTLY even if every code default drifts (user directive). Confirmed gap: the h5 saves config_yaml / train_args_yaml (search-collapsed, NOT default-materialized) — omitted knobs (norm, scheduler kwargs, n_gates, gate_init, compile_mode, freeze_trunk, berhu knots, ema, thresholds...) are resolved inside build_specs / run_emulator and never persisted; reconstruction silently trusts frozen defaults. FIX = persist the consumed view (the banner directive extended to artifacts): (1) build_specs assembles exp.resolved_model — a serializable rebuild recipe (class qualname, dims, every constructor kwarg with factories serialized by name: activation {type,n_gates}, norm, head pin) built BESIDE the specs by the same code; (2) run_emulator returns resolved_train (every training knob defaults-materialized incl. per-phase resolved blocks + derived eval_bs) — declared interface change; (3) save_emulator v2: keeps the raw configs (provenance of what was WRITTEN) + adds config_resolved_yaml + model_recipe/ group + attrs schema_version=2 / git commit best-effort; (4) NEW rebuild_emulator(path_root, device) in results.py — reconstructs module+geometries from the h5 ONLY (a missing recipe key = loud error, NEVER a code default), loads the .emul, returns the inference-ready triple; (5) THE acceptance property: save -> rebuild -> bitwise-equal outputs on a probe batch, AND the drift test — monkeypatch code defaults, rebuild unchanged. v1 files: rebuild_emulator refuses loudly (they predate the guarantee). Out of scope: training-resume checkpointing (optimizer state). Gates GSV-A..C. SEQUENCED AFTER the NPCE unit (same files). NOT implemented."
metadata:
  node_type: memory
  type: project
---

# Save schema v2: persist the consumed view (spec)

User directive 2026-07-07: "emul + h5 needs allow us to reconstruct
the emulator perfectly even if ALL default values of parameters
drift in the code over time."

Confirmed gap (results.py read): the h5 persists the geometry states
(solid), the histories, and config_yaml / train_args_yaml — the
driver's config with search ranges collapsed but DEFAULTS NOT
MATERIALIZED. Everything the YAML omitted is resolved at build/run
time and lost. The fix is the banner directive's second half:
displays render the consumed view; ARTIFACTS PERSIST IT — by the
same code that consumes, never a parallel re-implementation.

## Scope

1. exp.resolved_model (build_specs, assembled BESIDE the spec dicts
   so it cannot diverge): a fully serializable rebuild recipe —
   model class qualname + (name, ia, arch), input_dim / output_dim,
   every constructor kwarg actually passed, with callables
   serialized by name: activation {type, n_gates}, norm (the
   make_norm name), the head activation pin {type, n_gates} | None,
   mlp width / n_blocks, the active head's knobs, compile_mode as
   resolved, n_amps / n_templates for factored designs, needs_geom.
2. run_emulator returns resolved_train (DECLARED interface change:
   one added return, experiment.train stores it): assembled right
   after the up-front validation/defaulting — bs, nepochs, seed,
   thresholds, use_amp, clip, rewind, trunk_epochs, freeze_trunk
   (resolved bool), loss (defaults-filled incl. berhu knots +
   anneal | None), ema (resolved | None), lr {lr_base, bs_base,
   warmup_epochs, the computed lr}, optimizer {cls name,
   weight_decay, extras}, scheduler {cls name + kwargs}, trim /
   focus (defaults-filled), the per-phase blocks AS RESOLVED PER
   PASS, the derived eval_bs, device string.
3. save_emulator schema v2: KEEP config_yaml + train_args_yaml (the
   provenance of what the user WROTE) and ADD:
   - config_resolved_yaml: resolved_train + the resolved data block
     (n_train, n_val, split_seed, ram_frac, param_cuts filled);
   - model_recipe/ group: exp.resolved_model;
   - root attrs: schema_version = 2, code git commit (best-effort:
     rev-parse if a repo, else "unknown"), torch version + created
     (already there).
   The pce group ([[npce-yaml-wiring]]) joins config_resolved
   coverage when both land.
4. NEW rebuild_emulator(path_root, device) in results.py — the
   loader that PROVES the guarantee and the missing in-package
   inference entry point: reads model_recipe + the geometry states
   (+ pce when present), re-makes factories from names
   (make_activation / make_norm), instantiates the class, loads the
   .emul state_dict (strict=True), returns (model, param_geometry,
   geometry). HARD RULE: it reads ONLY the h5 — a missing recipe
   key is a loud error naming it; it NEVER falls back to a code
   default (that is the whole point). v1 files (no schema_version):
   refuse with a clear message (they predate the guarantee).
5. Out of scope, stated: training-resume checkpointing (optimizer
   moments, schedulers mid-run) — a different feature; this unit is
   inference reconstruction + provenance.
6. Docs: the h5 layout docstring (results.py) rewritten; the
   train_single driver header --save paragraph; the code map
   results.py entry (+ rebuild_emulator line); README Run-it
   outputs sentence gains "the h5 carries the fully-resolved config
   (schema v2) — a saved emulator rebuilds bit-exactly even if code
   defaults later change".

## Gates

- GSV-A (Mac, pure/stub): a completeness CENSUS — every run_emulator
  signature knob and every model constructor kwarg appears in
  resolved_train / resolved_model (diff the signatures against the
  dicts programmatically, so a future knob that forgets to join the
  recipe FAILS this gate); YAML serialization round-trip of both
  dicts; rebuild_emulator (stubbed io) refuses on a missing key and
  on schema_version < 2.
- GSV-B (Mac, static): resolved_model assembled beside the specs
  (line-ordered, same values, no parallel resolution); the
  interface changes declared; save schema groups/attrs; docs;
  scans + py_compile.
- GSV-C (workstation, rides the queue): train small -> save ->
  rebuild_emulator -> outputs BITWISE-EQUAL to the live model on a
  probe batch; THE DRIFT TEST: monkeypatch a handful of code
  defaults (make_scheduler patience, make_activation n_gates,
  make_model width, the norm default) and rebuild again — still
  bitwise-equal (proof the h5, not the code, defines the emulator);
  a v1 file refused with the clear message.

## Handoff

### ARCHITECT_HANDOFF
Task: save schema v2 — persist the consumed view (spec:
notes/save-schema-resolved-config.md in full; the HARD RULE is
binding: rebuild_emulator reads only the h5, never a code default).
Base: the NPCE commit; `git log -1` must show it — else STOP.
Scope items 1-6 exactly; interface changes (run_emulator return,
save_emulator args, new rebuild_emulator) declared in the report.
Gates GSV-A/B on the Mac; GSV-C embedded for the workstation queue.
Report: IMPLEMENTER_HANDOFF + resume state appended here, raw gate
outputs, deviations declared. Do not commit: leave the diff
uncommitted and print the suggested commit command.
### END

## Status

SPEC DELIVERED 2026-07-07, NOT implemented. SEQUENCED AFTER the
NPCE unit (both touch results.py / experiment.py). Suggested commit
sentence: "Save schema v2: the h5 persists the fully-resolved
consumed config + a serializable model recipe; rebuild_emulator
reconstructs bit-exactly from the file alone — immune to code
default drift (gates GSV-A/B Architect-verified)".

## STANDING RULE (user, 2026-07-07): artifacts never trust defaults

Verbatim: "The philosophy over the emul and h5 file has to be —
dont trust on default values — they can drift."

The two halves, binding on THIS unit and on every future
save/load surface (the pce group, any future checkpointing, every
new knob):

- WRITE side: everything the run consumed is written, defaults
  MATERIALIZED at save time — the file records values, never the
  fact that a value was defaulted. Enforced mechanically by the
  GSV-A census (signatures diffed against the recipe: a future
  knob that skips the recipe fails the gate).
- READ side: reconstruction reads ONLY the file — a missing key is
  a loud error naming it, NEVER a fallback to a code default.
  Enforced by rebuild_emulator's hard rule + the GSV-C drift test
  (monkeypatched defaults, bitwise-equal rebuild).

This is the consumed-view doctrine's second half
([[banner-prints-consumed-view]]): displays RENDER the consumed
view; artifacts PERSIST it; loaders TRUST ONLY it. Every future
spec that adds a knob or a save surface must state how it honors
both halves, and every audit checks them.

## Rider (2026-07-07): the emultrfv2 install-path sweep

User: the repo will be cloned into Cocoa as
external_modules/code/emulators/emultrfv2/ (no /dev segment; the
old documented path emulators/emultrf/dev is retired). FIFTEEN
doc-only sites (enumerated by grep): README.md:62 (the Run-it D=
line), emulator/README.md:67 (the code-map layout note), and twelve
driver-header comment lines across the five drivers (the
"python .../emultrf/dev/..." examples and the "sits beside the
emulator/ package (same .../emultrf/dev/ folder)" notes). Replace
every emultrf/dev with emultrfv2. All comment/doc lines — AST
code-identity must hold on the drivers. Gate (joins GSV-B): grep
"emultrf/dev" over the tree (notes/ exempt) returns ZERO;
grep emultrfv2 hits the same fifteen sites.

## Rider 2 (2026-07-07): the .paramnames cross-check (naming integrity)

User closed the naming loop: the dumps ship a getdist-style
<train_params>.paramnames whose FIRST column is the cobaya name of
each .txt column (trailing chi2* = the derived chi2 the staging
slice already drops) — so ParamGeometry.names are cobaya names by
construction, and get_requirements speaks the sampler's language
natively (the lambda bridge is only for true reparametrizations).

The integrity gap: TWO name sources exist — read_param_names reads
the COVMAT HEADER; the .paramnames sidecar independently declares
the .txt columns. They agree today by generator construction; a
divergence would silently pair wrong columns with wrong covmat rows
in the whitening. RIDER: at staging, WHEN a .paramnames file sits
beside train_params, cross-check its first column (minus the
starred derived entries) against the covmat-header names, order
included — mismatch = a loud error naming both lists; absent
sidecar = no check (back-compatible). One pure helper + one call in
load_source/experiment staging; a GSV-A case each (match, mismatch,
absent, starred-tail handling).
