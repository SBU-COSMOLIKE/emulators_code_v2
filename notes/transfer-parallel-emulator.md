# Transfer learning by a parallel correction emulator (spec)

**Date:** 2026-07-10. **Status:** SPEC (Architect, Fable). **Spec code:**
TPE. **Home note** for the gates `transfer-identity` / `transfer-smoke`.

## The request (user design, decisions already made in discussion)

A trained full emulator (say LCDM; for NLA/TATT the factored design whose
output is per-template) is FROZEN ENTIRELY. A new, smaller PARALLEL
emulator takes the FULL new parameter space (say EDE + w(z)-PCA on top of
LCDM) and outputs per-template corrections. The final data vector combines
the two element-wise, per template, BEFORE the amplitude combine — e.g.
`gg = gg1 * (1 + gg2)` — and the combination rule is a class picked by a
YAML option. The parallel net needs far fewer parameters because it only
expands the base.

**Why not fine-tuning (FTW):** the science target is model extensions of
~12 new parameters carrying new physics sectors (Early Dark Energy +
w(z)-PCA amplitudes), where adapting a same-capacity network is
structurally insufficient. FTW remains the tool for few-extras,
same-physics moves; TPE is the designed path for the real program.

**Settled in discussion:** the SPACE the composition acts in (physical
squeezed bins vs whitened coordinates) is a USER FLAG, orthogonal to the
form — the same code base will emulate galaxy-galaxy lensing, where gamma_t
crosses zero, so `sum` is first-class (not a fallback) and the user must be
able to test any form in either space. The spec records recommended
pairings and defaults; the user ultimately decides per study:

- `gain` recommends PHYSICAL: fractional corrections are O(1) and
  self-normalized there, and element-wise operations are physically
  meaningful in bin space; in whitened space the coordinates cross zero
  everywhere (the rotation mixes bins), so the near-zero degeneracy is
  generic rather than localized.
- `sum` recommends WHITENED: the residual inherits the exact
  base-cancellation metric, and the net output stays unit-variance; in
  physical space the additive output spans the decades whitening exists to
  tame. (The base cancellation is exact in BOTH spaces — the residual is
  linear — so sum/physical is a legitimate conditioning experiment.)

The pure product `gg1 * gg2` is dropped: it admits no zero-init identity
start.

## The pattern already exists twice — reuse it

- `PCERatioChi2` (losses/pce.py) IS the gain form over a frozen
  closed-form base: `pred = b * (1 + delta)` in physical squeezed space,
  division-free, with the SPEED DESIGN to copy verbatim — the frozen base
  is precomputed once per row at encode time and packed beside the truth
  into a wider staged target (`target_dim`, read by batching.py), so the
  hot loop never re-runs the base and backward passes touch only the small
  net.
- `PCEResidualChi2` IS the sum form: whitened residual, the base cancels
  in `(pred - target)`, metric exact.
- `TemplateFactoredChi2` (losses/ia.py) owns the insertion point: the
  model outputs whitened templates `(B, T, n_keep)`, `coeff_fn` combines
  them linearly (nla_coeffs T=3, tatt_coeffs T=10; IA_DESIGNS in
  experiment.py), amplitudes are imposed not learned. The correction
  composes template-by-template BEFORE that combine, so one corrected
  template set serves the entire amplitude family at zero coverage cost.

## Design rules

### D-TP1 — YAML surface

```yaml
transfer:
  from: projects/lsst_y1/emulators/lcdm_nla/emul_v2
  form: gain
  space: physical
```

- Top-level block (sibling of `pce:`, one base per study). `from` = the
  frozen base artifact path root, resolved exactly like `finetune.from`
  (expanduser/expandvars, relative joins $ROOTDIR, both files must exist).
- `form` = the combination rule: `gain` (`base * (1 + r)`) or `sum`
  (`base + r`). Unknown form -> loud error listing the registry.
- `space` = where the composition acts: `physical` (squeezed bins) or
  `whitened`. OPTIONAL; absent -> the form's recommended space (gain ->
  physical, sum -> whitened), materialized into the resolved config and
  the artifact (persist-resolved-values: the file records the space
  actually used, never "it was defaulted"). Choosing the non-recommended
  pairing is ALLOWED — the user decides — and prints one quiet-gated
  notice line naming the trade-off (the activation-flag-notice precedent):
  gain/whitened = the near-zero degeneracy is generic (whitened
  coordinates cross zero everywhere); sum/physical = the net output spans
  the decades whitening exists to tame. Neither is an error.
- Exclusivities, all loud at config time: `transfer:` with `pce:`; with
  `--rescale` other than none; with `train_args.finetune`; with a
  `model.ia` key (the family is inherited, D-TP2); two-phase keys follow
  the finetune rule (single-phase V1).
- The `model:` block IS present and describes the CORRECTION net only —
  capacity (width / n_blocks / activation) is the user's knob here, unlike
  FTW. Its family is forced by the base (D-TP2).

### D-TP2 — the frozen base: loaded like FTW, family inherited

Load once via `rebuild_emulator(compile_model=False)` (the warmstart.py
loader path; cache on the experiment). Constraints, loud errors:

- schema v2; source `rescale` attr `"none"` (and the run's --rescale
  none); no `pce` group; no `transfer_base` group (NO CHAINING, V1);
- allowed input geometries: plain `ParamGeometry` AND
  `AmplitudeFactorGeometry` — the factored NLA/TATT base is the headline
  use, so this unit LIFTS the FTW D-FT10 factored exclusion for its
  loader (LogParamGeometry stays out, V1);
- the correction net's family is INHERITED from the base:
  base plain -> plain correction (output n_keep); base ia = nla/tatt ->
  template correction with the SAME n_templates (TemplateMLP machinery,
  no new architecture code). A YAML `model.ia` or a name that changes the
  family is a loud error naming the base's family.

### D-TP3 — input geometry: the FTW block extension, generalized

The run's parameter space is a superset of the base's (the extras carry
the new physics; n_x may be ~12). Reuse `warmstart.extend_input_geometry`
math (D-FT3) with one generalization:

- plain base: identical to FTW — extended plain ParamGeometry, shared
  coords bit-identical to the base encoding, extras appended last.
- factored base (AmplitudeFactorGeometry): extend the INNER `pg_keep`
  block by the same math; the amplitude columns stay raw and LAST (the
  TemplateFactoredChi2 slicing contract). Encoded layout:
  `[shared-whitened (n_s') ; extras-whitened (n_x) ; raw amps (n_amp)]`.

**The invariant that makes the base evaluable with zero extra plumbing:**
because the shared coordinates are bit-identical to the base's own
encoding, the base's input is a COLUMN SLICE of the run's encoding —
`cat(enc[:, :n_s'], enc[:, -n_amp:])` for a factored base, `enc[:, :n_s]`
for a plain one. No second geometry evaluation, no raw-theta plumbing into
the loss; the D-FT3 invariant IS the interface. The gates assert this
slice reproduces the base's own encoding bitwise.

Output geometry: pinned from the base per D-FT4 verbatim (same dataset /
mask / probe checks; the pinned center persists again).

### D-TP4 — the combination classes (emulator/losses/transfer.py)

The registry is the form x space matrix — four combinations, every one
identity-start-exact, implemented as classes mirroring the PCE pair (or
one family parameterized by form/space/T; Implementer's choice, guided by
the existing four). The shared skeleton for all: encode runs the frozen
base ONCE per row on the D-TP3 slice, converts it to the chosen space,
and packs `[base ; truth]` into the wider staged target (target_dim;
2*n_keep plain, (T+1)*n_keep factored); the hot chi2 unpacks and composes,
never re-running the base.

- `gain/physical` (the recommended gain): exactly PCERatioChi2 with the
  frozen neural base — hot path `b * (1 + pred) - xi`, decode
  `b * (1 + pred)`. Factored: base templates converted per template to the
  physical representation consistently with TemplateFactoredChi2's center
  handling (the center is absorbed in the constant-coefficient GG
  template; the combine commutes with the linear unwhiten), corrected
  `K_t <- K_t * (1 + r_t)` BEFORE the coeff_fn combine.
- `sum/whitened` (the recommended sum): the residual form; the base
  cancels in `(pred - target)` — metric exact, PCEResidualChi2's proof
  carries. Factored: per-template residual on the whitened templates.
- `gain/whitened` and `sum/physical`: the same skeleton with the space
  swapped — mathematically well-defined, allowed under the D-TP1 flag,
  each carrying its documented conditioning trade-off. The sum residual
  cancellation is exact in both spaces (linearity); gain's identity is
  exact in both (`x * (1 + 0) == x`).

**The identity invariant defines correctness for every form:** with the
correction output identically zero, the composed prediction must equal
the frozen base's own decode BITWISE (`1 + 0 == 1` exactly; the base path
is literally the same computation). The pre-train parity gate demands
`torch.equal` — stronger than FTW's tolerance, and any center/whitening
mishandling in the factored-gain plumbing breaks it loudly.

### D-TP5 — zero-init identity start

After make_model builds the correction net, the transfer path zeroes the
final output Linear (weight AND bias) before training — state-surgery in
the builder (the FTW transfer_state_dict precedent), not an architecture
option. Epoch 0 is then exactly the frozen base; the parity gate checks
it; the loop's incoming-weights snapshot makes it the epoch-0 baseline.
Optimizer / scheduler / EMA / trim / focus: all the normal fresh
machinery; lr is the ordinary `lr:` block (no new knob).

### D-TP6 — artifact: embed the base, never reference it

save_emulator gains a `transfer_base` group (the `pce` group precedent):
the base's recipe, state dict, both geometry states, and the `form` attr —
the saved transfer artifact is fully self-contained and survives the base
file moving. rebuild_emulator returns `info["transfer_base"]` /
`info["transfer_form"]`; the predictor composes base + correction on the
same slice contract; the cobaya adapter follows the existing pce_base
pattern. Provenance root attrs: `transfer_from` (resolved absolute root),
`transfer_form`, `transfer_extra_names` (space-joined). Chaining is
refused at load (D-TP2).

### D-TP7 — experiment wiring (the FTW branch pattern)

`validate_transfer` beside validate_pce (exclusivities above);
build_geometry branch: extend + pin via warmstart functions (plus the
AmplitudeFactorGeometry extension); build_specs: correction model from the
YAML model block, family/n_templates forced from the base info;
train: zero-init surgery + bitwise parity gate (one verdict line,
essential-only) + the normal loop. print_design: one banner line naming
the base, the form, and the extras. resolved config records the resolved
transfer block (persist-resolved-values).

### D-TP8 — validation gates

**`transfer-identity` (TPE-A; Mac + board; torch, no cosmolike).**
Synthetic PLAIN base and synthetic FACTORED (nla-like, T=3) base, both
through save_emulator round-trips; assert for ALL FOUR form x space
combinations (gain/sum x physical/whitened):
1. the D-TP3 slice reproduces the base's own encoding bitwise (plain and
   factored, extras interleaved in the covmat order);
2. epoch-0 identity: composed prediction `torch.equal` the base's decode,
   and independent of the extras bitwise;
3. target packing/unpacking exactness (the hot path never re-runs the
   base: assert by call-counting the base forward);
4. zero-init surgery: final-layer tensors exactly zero, every other
   tensor untouched;
5. error paths: chaining refused; pce/rescale/ia exclusivities; unknown
   form lists the registry; non-superset names.

**`transfer-smoke` (TPE-B; board, save-and-sample tier,
deps=("save-rebuild-drift",)).** A names-equal gain-form 2-epoch run over
the board's frozen plain base (`gates_emul_evaluate`): parity verdict
line, banner, run completes, artifact saved; the `transfer_from` /
embedded-base confirmation is the workstation leg (the FTW deviation-(b)
pattern). Honest margin recorded now: the factored-base and the n_x > 0
legs ride real data only when an NLA base artifact and an extended dump
exist — the Mac gate owns both until then.

### D-TP9 — out of scope (V2, recorded)

**Joint refinement / alternation (user question 2026-07-10, ruled with
rationale; cost model corrected by the user same day):** the metric is
SAMPLE EFFICIENCY — accuracy per training cosmology
([[emulator-sample-efficiency-is-the-goal]]) — never wall-clock; extended
-model data vectors (EDE + w(z)-PCA, new-experiment accuracy flags) are
the expensive object, so the compute cost of unfreezing the base is NOT a
design constraint (the packed-target optimization simply switches to live
base evaluation in any unfrozen stage — a mode, not a redesign). Under
that metric the frozen-base default is favored MORE strongly, for a
different reason: freezing transfers information from the cheap LCDM dump
to the expensive extended dump, and unfreezing re-exposes the full
parameter count to exactly the few rows that cannot be afforded again
(overfitting risk grows in the regime that matters). N-cycle alternation
IS a crude trust region (frozen most of the time), but it parameterizes
base drift through a discrete many-knob schedule (N, per-phase epochs,
LRs) where each tried setting burns a full run on the same scarce rows.
The V2 kernel keeps the user's secondary LR and sharpens the trust
region: ONE optional joint-refinement stage after the correction
converges — base unfrozen at a per-group secondary LR (make_optimizer
extension; the two-phase engine with freeze_trunk: false is the
structural precedent) PLUS an optional anchor penalty
lambda * ||W_base - W_base_pretrained||^2 (L2-SP, the transfer-learning
standard: lambda -> inf recovers the frozen base, lambda -> 0 is free
fine-tuning, a continuous dial instead of a schedule), with base-drift
diagnostics (delta-weight norm + before/after eval on the base's own
validation slice) as first-class artifact records. Deciding experiment,
in the right currency: fixed expensive-dump N_train, race frozen-only vs
joint+anchor at a few lambdas. Alternation earns its way in only if
anchored-joint shows instabilities that scheduling demonstrably cures.

Chaining (transfer over a transfer artifact); cross-family composition
(NLA base + the extra TATT templates added additively — the natural next
step for the TATT program); NPCE composition; cross-dataset/mask transfer;
LoRA-style weight deltas (rejected: the parallel-output correction is a
physical object — the per-template fractional correction is directly
plottable against the new parameters as a science diagnostic — and the
composition machinery already exists).

### D-TP10 — the joint refinement stage (designed 2026-07-10; UNIT 2)

User-endorsed same day (promoting the D-TP9 kernel to an offered
feature): the ULMFiT-style two-stage plan. Stage 1 = TPE V1 exactly as
specced (correction warms, base frozen, packed targets). Stage 2,
OPTIONAL = unfreeze ONCE and train jointly with discriminative learning
rates plus the L2-SP anchor. Executes as its own unit AFTER TPE V1 gates
green (it builds on V1's composed model and losses; unit-size discipline
per the FTW delta lessons).

```yaml
transfer:
  from: projects/lsst_y1/emulators/lcdm_nla/emul_v2
  form: gain
  space: physical
  refine:
    epochs: 200
    base_lr_scale: 0.01
    anchor: 1.0e-2
```

- `refine:` absent = frozen-only (V1 behavior, byte-identical).
- `epochs` = stage-2 epochs, appended after the correction stage.
- `base_lr_scale` = the base parameter group's LR as a multiplier on the
  run's resolved learning rate (the correction group keeps the full
  value). Named to avoid colliding with lr.lr_base (the sqrt-batch
  baseline — an unrelated meaning of "base").
- `anchor` = the L2-SP lambda on the base group,
  lambda * ||W_base - W_base_pretrained||^2, applied during stage 2
  only, as a DECOUPLED post-step update (the shared anchor facility
  defined in finetune-warm-start.md's anchor extension: AdamW-style
  decoupling keeps the anchor out of Adam's moments; set
  weight_decay 0.0 beside it, documented there). REQUIRED when refine: is present — an explicit 0.0 states
  "free fine-tuning" deliberately; there is no silent default (the
  never-trust-defaults rule applied to a physics-consequential knob).
- Mechanics at goal level: stage 2 starts from stage 1's best-restored
  weights (loss-continuous trivially — same weights); fresh optimizer
  with per-module parameter groups (a make_optimizer extension: base
  group at scaled LR + anchor, correction group normal; the decay split
  applies within each); packed targets switch to live base evaluation
  (the base re-enters the graph — a loss mode, not a redesign; compute is
  explicitly not a constraint, D-TP9); the existing two-phase engine
  (freeze_trunk: false joint mode, per-phase scheduler/warmup/best-
  restore) is the structural precedent.
- Diagnostics, first-class in the artifact: the base's drift (relative
  delta-weight norm, total + per-layer) and the correction-magnitude
  stats, plus a banner line; the saved transfer_base group keeps the
  PRETRAINED base (the anchor's reference and the provenance object),
  with the drifted base weights in the main state dict.
- The deciding experiment (D-TP9) stands: fixed expensive-dump N_train,
  frozen-only vs refine at a few lambdas.

## Links

[[finetune-warm-start]] (the loader + block-extension this reuses; D-FT3/
D-FT4 are load-bearing here), [[designs-losses-family-folders]] (where
losses/transfer.py sits), [[py-module-style-conventions]],
[[gates-checks-docs-plain-language]], [[gates-harness-user-run]].

## Resume state (Implementer appends below)

### TPE V1 execution (2026-07-10, Opus) — CORE + ACCEPTANCE GATE DONE; artifact lifecycle REMAINS

Base: main 18f7207 (FTW closed, board 22/22). Diffs uncommitted on branch
claude/amazing-keller-e798b6. The user runs git. Scope guard honored: D-TP1..
D-TP9 only; refine:/anchor are loud not-yet-implemented (unit 2).

**Done + Mac-verified (compileall + AST + numpy probes; torch gate deferred to a
torch machine, Mac has no torch):**
- `emulator/losses/transfer.py` (new, 451): `TransferChi2`, one parameterized
  family over form(gain/sum) x space(physical/whitened) x {plain, factored}.
  Packed-target skeleton (`target_dim` 2*n_keep plain / (T+1)*n_keep factored;
  base run once at `encode`, unpacked in the hot chi2). `_base_input` is the
  D-TP3 column slice; `_compose` (gain/sum); `_combine` (coeff_fn); physical /
  whitened metric both reduce to `r_phys^T Cinv_sq r_phys`. `base_decode`
  returns the frozen base in the loss's own space (the parity reference).
  FORMS / SPACES / RECOMMENDED_SPACE exported.
- `emulator/warmstart.py`: `extend_input_geometry` generalized — a plain base
  extends directly (`_extend_param_geometry` extracted), a factored base
  (AmplitudeFactorGeometry) extends its inner pg_keep with the amps kept last
  (D-TP3). `load_source(allow_factored=True)` lifts the FTW factored exclusion
  (Log stays out), rejects a `transfer_base` group (no chaining, D-TP2), and
  returns `.ia`. `build_transfer_start` (zero-init the correction's final
  Linear via `_zero_final_linear` + the bitwise parity gate: epoch 0 ==
  frozen base + extras-independence, D-TP5/D-TP7). `validate_finetune_config`
  rejects `anchor` (unit 2).
- `emulator/experiment.py`: `validate_transfer` (D-TP1/D-TP2 exclusivities +
  form/space resolution with the off-recommendation notice + refine
  not-implemented); from_config / build_geometry / train / print_design
  transfer branches (correction family forced from the base; build_specs needs
  NO branch — the correction has a real model block and the normal path forces
  n_templates/n_amps from self.ia). resolved_train records the transfer block.
- `gates/checks/transfer_identity.py` (new, 415): TPE-A, synthetic plain +
  factored bases through save_emulator round-trips, all 4 form x space:
  slice==base encoding (bitwise), epoch-0 identity via build_transfer_start,
  base-cached (forward-hook call count), target_dim, zero-init surgery, and
  the config error paths. `gates/board.py` gate_tpe_a + gate_tpe_b + 2 BOARD
  rows (identity new-features torch; smoke save-and-sample deps
  save-rebuild-drift). `gates/board_config.json` + `gates/configs/
  transfer-smoke-config.yaml`.

**KEY design finding (flagged for the Architect):** the factored-PHYSICAL base
representation is per-template un-whitened, `combine(unwhiten(K))`; the standard
factored decode is `unwhiten(combine(K))`. These are equal physics but differ
by float REASSOCIATION (~4e-6 in the numpy probe, `combine`/`unwhiten` do not
commute bitwise). So the epoch-0 identity is bitwise against the base evaluated
in the composition's OWN space (`base_decode()`), which is what the gate
compares — NOT against the standard whitened-combine decode (~4e-6 off for
factored-physical only; the other 6 combos are bitwise against both). This
satisfies "epoch 0 IS the frozen base, the base path is literally the same
computation" (D-TP4). If the Architect wants torch.equal against the standard
decode for factored-physical too, the space semantics need a rethink.

**REMAINING (the save / reload / predict lifecycle; specified, not yet coded):**
1. `emulator/results.py`: save_emulator gains a `transfer_base` group (the pce
   group precedent) holding the base recipe (yaml), base state_dict, both base
   geometry states + cls markers, and form/space attrs; rebuild_emulator reads
   it back (reconstruct base model from the embedded recipe + load state +
   rebuild both geometries) and returns info["transfer_base"] /
   info["transfer_form"] / info["transfer_space"]. Factor the existing main
   recipe->model reconstruction into a nested helper and reuse it for the base.
2. `emulator/inference.py`: the composed predictor (pce_base pattern) —
   rebuild returns the base bundle + form/space; predict composes base +
   correction on the same slice contract (TransferChi2.decode logic at
   inference). The cobaya adapter follows the pce_base branch.
3. `train_single_...py`: assemble the transfer_base dict from exp._transfer_base
   and pass it to save_emulator; add root attrs transfer_from / transfer_form /
   transfer_extra_names (only on a transfer run). Enumerate every cfg-key
   access on the driver path first (FTW lesson 1): a transfer YAML HAS a model
   block, so the FTW missing-model bug does not recur, but check run_tag and
   any model.ia read.
4. `example_yamls/transfer_emulator_cosmic_shear.yaml` (new) + a pointer in the
   train_single YAML.
Until 1-3 land, a transfer run TRAINS + parity-checks correctly but does not
save a reloadable artifact (transfer-smoke's save leg + the artifact round-trip
need them). The transfer-identity Mac gate does NOT depend on 1-4 (it builds
TransferChi2 directly).

## Architect audit verdict (TPE V1, 2026-07-10, Fable)

**VERDICT: VERIFIED for the delivered scope — sign-off given, with one
Architect delta applied (D-TPE-1) and two rulings.**

- **Ruling 1 (the factored-physical finding):** the bitwise epoch-0
  identity is against the SPACE-MATCHED base representation
  (base_decode()) — that is the invariant that proves the plumbing. The
  ~4e-6 vs the standard whitened-combine decode is float REASSOCIATION
  (combine and unwhiten do not commute bitwise), expected mathematics:
  keep it as a documented tolerance leg (<= 1e-5), never bitwise.
- **Ruling 2 (partial unit): APPROVED** — the core + acceptance gate land
  now; TPE-1b (results.py transfer_base embed, inference.py composed
  predictor, driver provenance, example YAML — fully specified in the
  resume above) is the NEXT handoff and must land before any science use.
- **Delta D-TPE-1 (Architect, applied directly, compile-checked):** the
  driver save was unconditional, so a completed transfer run would have
  persisted the CORRECTION NET ALONE wearing the normal artifact schema —
  reloadable and silently wrong. The driver now refuses the save on a
  transfer run with one loud line and a clean exit; transfer-smoke gains
  the assertion that the refusal printed (replaced in TPE-1b by the real
  save + provenance legs).
- **Evidence:** load-bearing spans read line-by-line (_base_input slice,
  _compose, _unwhiten_templates center convention, validate_transfer
  error paths incl. refine/anchor unit-2 guards); the Implementer's
  8-combo numpy probe; the transfer-identity torch gate is the
  workstation confirmation (the FTW evidence pattern — no torch on
  either Mac).

**Next:** user lands the branch; board runs transfer-identity +
transfer-smoke; then the TPE-1b handoff.

## TPE V1 closure (2026-07-10): board 24/24, both gates PASS first try

**transfer-identity (TPE-A): PASS on CUDA, 36/36** — all eight
form x space x family identity combos bitwise, the slice contract bitwise
for plain and factored, base-cached proven by call-counting (encode 1,
chi2 +0), zero-init surgery exact, all seven error paths including the
unit-2 refine guard. **transfer-smoke (TPE-B): PASS** — parity bitwise on
200 rows, banner names base + form gain/physical, the 25,927-param
correction trains over the frozen 66k base (val 23208 -> 22632 in 2
epochs at full lr — the correction moves freely, the base bit-frozen),
and the D-TPE-1 refusal line printed + asserted: no correction-only
artifact exists. Whole board green (24 rows incl. triangle-shading).
**Next: the TPE-1b handoff** (the artifact lifecycle specified in the
resume above); then unit 2 (refine + the shared anchor facility).

## TPE-1b execution (2026-07-10, Opus) — the artifact lifecycle, coded

Base: main 33a3e64 (TPE V1 closed, board 24/24). Diffs uncommitted on branch
claude/amazing-keller-e798b6. The user runs git. This landed REMAINING 1-4
from the TPE V1 resume plus the driver-refusal replacement and the smoke
round-trip leg.

**Done + Mac-verified (compileall + AST; the artifact code needs torch + h5py,
absent on this Mac, so the round-trip runs on the board):**
- `emulator/results.py`: `save_emulator` gains a `transfer_base=None` param
  writing a `transfer_base` group (base model_recipe yaml + a `state/`
  subgroup of the base weights + `param_geometry/` + `dv_geometry/` with cls
  markers + form/space attrs). The recipe->model reconstruction in
  `rebuild_emulator` is factored into a nested `_rebuild_model(rc, geom, state,
  want_compile)` reused for the main model (the correction) and the base;
  rebuild reads the group back (base rebuilt eager, never compiled) and returns
  `info["transfer_base"]` (model/pgeom/geom) + `["transfer_form"]` +
  `["transfer_space"]`. The main model is the CORRECTION, the main geometries
  the run's; the artifact is self-contained (no base path reference, D-TP6).
- `emulator/inference.py`: `_build_decoder` gains a transfer branch (FIRST, so
  it wins over the ia branch a factored correction would otherwise take): it
  builds a `TransferChi2` from the embedded base + form/space (family read off
  the base geometry, coeff_fn/T from the correction's inherited `ia`) and
  returns `chi2.decode` — the exact training composition, single-sourced. The
  cobaya adapter needs NO change (thin shell over `predict`, composes
  transparently); a sampling YAML must sample the extras (the existing
  dependency-resolution error covers it, the FTW precedent).
- `train_single_...py`: the D-TPE-1 refusal guard is REPLACED by the real save
  — assemble the `transfer_base` dict from `exp._transfer_base` (recipe, state,
  both geometries, form/space) + provenance root attrs `transfer_from` /
  `transfer_form` / `transfer_space` / `transfer_extra_names` (transfer run
  only). FTW cfg-key sweep done: the only driver-path `cfg[...]` reads are
  `cfg["data"]["train_dv"/"val_dv"]` + `cfg["data"].get("param_cuts")` (a
  transfer YAML has a data block) and `run_tag` uses `exp.arch` (set in the
  transfer from_config branch), so no missing-key bug.
- `gates/checks/transfer_identity.py`: new `check_lifecycle` leg — save a
  transfer artifact (correction + embedded base), rebuild, and assert the
  composed prediction reproduces the in-memory composition BITWISE both via
  `rebuild_emulator` + the transfer decoder and via the full `EmulatorPredictor`
  path (the save-rebuild-drift pattern), plus chaining refused (load_source on
  the saved transfer raises). `gates/board.py` transfer-smoke: the refusal
  assertion is replaced by the save-completed legs (the two `saved ...` lines)
  + a logged provenance/round-trip follow-up.
- `example_yamls/transfer_emulator_cosmic_shear.yaml` (new) + a pointer in the
  train_single YAML.

**Gate reruns the user must force (both recorded PASS under the old asserts):**
`--force-rerun transfer-identity` (now has the lifecycle leg) AND
`--force-rerun transfer-smoke` (the refusal assertion is gone; it now asserts
the save completed). The Architect confirms the saved artifact's `transfer_from`
attr + embedded `transfer_base` group + rebuild round-trip on the workstation
(the FTW deviation-(b) pattern).

## Architect audit verdict (TPE-1b, 2026-07-10, Fable)

**VERDICT: VERIFIED, sign-off conditional on the two workstation
force-reruns going green** (the Mac has no torch/h5py, the established
evidence pattern). Read line-by-line: the driver assembles the
self-contained transfer_base bundle (recipe + state dict + both
geometries + form/space) and stamps all four provenance attrs;
results.py writes/reads the group symmetrically via the factored
_rebuild_model; the lifecycle leg asserts the strongest form of the
D-TP6 contract — save -> rebuild -> composed predict BITWISE vs the
in-memory composition, predictor bitwise, and chaining refused on the
REAL saved artifact (not a synthetic tweak). The cobaya adapter needing
no code (the predictor composes transparently) is consistent with the
pce_base precedent and accepted. No deviations. D-TPE-1's refusal guard
correctly replaced by the real save + assertions.

Workstation: `python gates/run_board.py --force-rerun transfer-identity`
then `--force-rerun transfer-smoke` (or one pass with both flags if
supported); after green, confirm the smoke's saved h5 attrs
(transfer_from / form / space / extra_names) from the `saved run record`
path — then TPE-1b closes and unit 2 (refine + shared anchor) is the
next handoff.

## TPE-1b closure (2026-07-10): board 24/24 again, lifecycle green

transfer-identity (with the new lifecycle leg: save -> rebuild ->
composed-predict bitwise, predictor bitwise, chaining refused on the real
artifact) and transfer-smoke (now SAVING: parity bitwise, banner,
`saved run record -> chains/emulator_resmlp_t16_ntrain200.h5`) both PASS
on the force-rerun. The composed run persists a fully self-contained
artifact. One item closes the unit: the Architect's one-command attrs
confirmation from the workstation artifact (transfer_from / form /
space / extra_names), below. After that: UNIT 2 (transfer.refine + the
shared L2-SP anchor facility + finetune.anchor) is the final
infrastructure handoff before the EDE + w(z)-PCA program is pure science.

**Attrs confirmation (2026-07-10, user-run):** transfer_from = the
absolute gates_emul_evaluate root, form/space = 'gain'/'physical',
extra_names = '' (names-equal), transfer_base group present — all exactly
as specified. **TPE-1b CLOSED.** Unit 2 (refine + shared anchor) is the
final infrastructure handoff.

## Architect ruling + delta re-handoff (TPE-2 checkpoint, 2026-07-10)

**RULING: the second-group refined-artifact layout is APPROVED and
supersedes D-TP10's literal "drifted base in the main state dict".**
transfer_base/ keeps the pretrained W0 (provenance + anchor reference,
V1-byte-identical); transfer_base/drifted_state/ + a transfer_refined
attr hold the drifted base; the correction stays the main model/.emul.
Rationale: preserves the main-.emul-is-the-recipe's-model invariant, a
stage-1-only artifact stays byte-identical to a V1 save (free regression
guard), and the drift is recomputable from the file alone (the
persist-resolved-values philosophy applied to a training trajectory).

**Conditions:** (1) two-way consistency, loud — transfer_refined true
iff drifted_state present, rebuild refuses either half alone; (2) the
lifecycle gate asserts BOTH the refined round-trip bitwise AND the
unrefined path byte-identical to a V1 save; (3) drift attrs (total +
per-layer relative norms) stamped from the stashed pretrained clone,
recomputable from the artifact; (4) the composed predictor picks
drifted-when-present silently — no flag, no fallback ambiguity.
Per-group-LR finding (warmup/scheduler preserve the group ratio):
endorsed. Full training-path audit rides the finished handback.
## TPE-2 execution (2026-07-10, Opus) — refine + the shared anchor facility

Base: main 7d34fc8 (TPE-1b closed). Diffs uncommitted on branch
claude/amazing-keller-e798b6. The user runs git. Read D-TP9/D-TP10 + the
finetune-warm-start.md anchor-extension section in full before starting.

### PROPOSAL for audit: the refined-artifact layout (the design-sensitive area)

D-TP10 says "transfer_base keeps the PRETRAINED base; the drifted base
weights in the main state dict." I propose the SECOND-GROUP variant of that
(the handoff offered it explicitly), because it keeps the whole V1
rebuild/predict path and .emul schema byte-identical:

- `transfer_base/` group: UNCHANGED from V1 = the PRETRAINED base (recipe +
  `state/` + both geometry states + form/space). It is the provenance object
  AND the anchor reference W_0. Self-contained (D-TP6) exactly as today.
- `transfer_base/drifted_state/` subgroup: NEW, present only on a refined
  run = the DRIFTED base weights (name -> tensor), plus a root attr
  `transfer_refined = True`. Absent on a frozen-only (V1) run, so a V1
  artifact is byte-identical.
- Main model + `.emul`: still the CORRECTION net alone (V1 schema unchanged).
- rebuild: build the base from `transfer_base`'s recipe; load `drifted_state`
  when present (the predictor composes correction + DRIFTED base), else the
  pretrained `state` (V1). The pretrained stays available for provenance +
  drift diagnostics.

Why not the drifted base literally in the main `.emul` under a `base.`
prefix: that forces rebuild to split correction./base. keys and build a
composite at inference, branching the V1 predictor path; the second-group
variant keeps `_rebuild_model(correction)` untouched and only adds "load the
base's drifted weights if present." The training-time composite (below) is
split on save, so the artifact layout stays decoupled from the training
model. The bitwise round-trip gate holds either way (float32-exact weight
round-trip); the refined-identity leg asserts it. **If the Architect prefers
the literal main-state-dict form, only the save-split + the rebuild-load
change; the training path and the anchor facility are identical.**

### The shared anchor facility (built once, both consumers)

Decoupled post-step update, never a loss term (the AdamW-decoupling argument):
after `optimizer.step()`, under no_grad, `W <- W - lr * lambda * mask *
(W - W_0)`, `lr` read per optimizer group. Mirrors the EMA post-step in-place
update already in training_loop_batched (same _foreach pattern, ~tens of us).
An `Anchor` object holds, per anchored param, (param, reference W_0, mask,
group index); `apply(optimizer)` does the one in-place step. weight_decay
decays toward 0 (away from W_0), so an anchored run should set weight_decay
0.0 (a quiet-gated notice, never an error).

- finetune.anchor: W_0 = the transferred init_state; mask excludes the padded
  extra columns (transfer_state_dict's padded_keys; those are exact-zero
  carriers of the new physics, anchoring them to 0 fights the warm start).
- refine.anchor: W_0 = the PRETRAINED base weights; mask = all-ones (the base
  has no padded columns); applied only to the base param group in stage 2.

### Status (Mac-limit: compileall + AST; torch + the numpy probes deferred)

**DONE + compile/AST-verified:**
- Shared anchor facility (training.py): `Anchor` (decoupled post-step
  `W <- W - lr*lambda*mask*(W-W_0)`, per-group lr, mirrors the EMA in-place
  update) + `build_anchor` + threaded through `training_loop_batched` and
  `run_emulator`. The loop's warmup + plateau scheduler were confirmed
  PER-GROUP safe, so discriminative LRs survive as preserved ratios.
- finetune.anchor: whitelist + validation (warmstart), `anchor_masks` (padded
  columns excluded via padded_keys), `build_warm_start` returns padded_keys,
  experiment builds the anchor spec + weight_decay-0 notice + passes anchor.
- Refine training path: transfer.py live-base mode (`set_live`, `_base` grad
  branch, `_base_in_space` / `_truth_half`, chi2 live branch; live=False
  byte-identical); `TransferComposite` + `make_refine_optimizer` (base @
  lr*base_lr_scale, correction @ lr, decay split within each); run_emulator
  `refine=` + the appended stage-2 pass (unfreeze base, capture pretrained
  W_0, composite, live loss, base-group anchor, train); validate_transfer
  refine {epochs, base_lr_scale, anchor REQUIRED} + materialization;
  experiment stashes `_transfer_pretrained_base` before drift + passes refine
  + the refine banner + resolved_train records the block.

**FINISHED after the Architect ruling (second-group layout APPROVED):**
1. Refined ARTIFACT (results.py): save writes the drifted base to a
   `transfer_base/drifted_state/` subgroup + a `transfer_refined` root attr;
   rebuild enforces the two-way consistency (attr present <=> group present,
   either half alone is a loud KeyError) and loads the DRIFTED base when
   present (else pretrained), so the predictor composes with the drifted base
   silently (no flag, condition 4). Driver: `state` = the pretrained clone,
   `drifted_state` = the drifted base, `transfer_refined` = True, plus the
   drift attrs `base_drift_total` + per-layer `base_drift_<name>` (relative
   norms, recomputable from the file's two states) + a banner line.
2. Gates: finetune_identity gains `check_anchor` (mask zeroes exactly the
   padded columns; the decoupled anchor pins the source columns, leaves the
   extras free, lambda 0 is a no-op). transfer_identity gains
   `check_refined_lifecycle` (a refined artifact save -> rebuild -> composed
   predict bitwise with the DRIFTED base + the two-way-consistency refusal).
   The unrefined byte-identical direction is the existing `check_lifecycle`
   (a stage-1 save writes no drifted_state / transfer_refined). A refine smoke
   leg on the board (2+2 epochs, tiny lambda, drift attrs) is the workstation
   follow-up (needs a real base + dump; the Mac gate owns the mechanism).
3. Example YAMLs: transfer.refine block + finetune.anchor key added
   (commented, with the weight_decay-0 recommendation). numpy probes:
   `tpe2_anchor_probe.py` (decoupled update pins/free + the padded mask) and
   the existing `tpe_loss_probe.py` (live-mode is additive) both all-green.

**Gate reruns the user must force:** `--force-rerun transfer-identity` (adds
`check_refined_lifecycle`) and `--force-rerun finetune-identity` (adds
`check_anchor`). Both are recorded PASS under the old assertions.

**Mac verification:** compileall (whole tree) + AST (run_emulator/loop accept
anchor+refine, all callers pass them, Anchor/TransferComposite/
make_refine_optimizer defined) + both numpy probes all-green + board loads 23.
The torch gate legs (check_anchor, check_refined_lifecycle, the refine
training) run on the workstation (Mac has no torch/h5py, the established
pattern).

## Architect audit verdict (TPE-2 final, 2026-07-10, Fable)

**VERDICT: VERIFIED, sign-off conditional on the workstation reruns**
(the established torch-blind-Mac pattern). Spot-audited at the three
load-bearing sites: the Anchor's decoupled step runs in place AFTER
optimizer.step() reading each group's current warmed/scheduled lr (the
AdamW-decoupling contract, per-group correct); results.py enforces the
two-way transfer_refined <-> drifted_state consistency with a loud
refusal on either half alone (ruling condition 1); anchor_masks builds
the finetune exclusion from padded_keys (the padded-columns rule). The
second-group layout is implemented per the checkpoint ruling; the two
Mac probes (decoupled pin/free + mask; live-mode additive) passed. The
refine-smoke board leg deferred to a real base+dump is accepted — the
Mac gates own the mechanism, the deciding D-TP9 experiment owns the
science.

**Workstation:** `--force-rerun transfer-identity` +
`--force-rerun finetune-identity` (both gained legs); the refine smoke
rides the first real science base. Green reruns close TPE-2 and the
INFRASTRUCTURE PROGRAM: scratch / anchored warm-start / frozen-base
transfer / anchored joint refinement — every path gated, every artifact
self-describing. The thread after this is science (D-TP9's deciding
experiment first).

**Delta D-TPE2-2 (2026-07-10, Architect):** transfer_identity still
carried the V1 "refine not-yet-implemented" error leg; TPE-2 implemented
refine, so validate_transfer now raises ValueError for the incomplete
block (base_lr_scale absent) — correct behavior, stale expectation. The
leg now asserts the TPE-2 rule: refine knobs are required-explicit (the
anchor precedent), an incomplete block is a loud ValueError. All 36
substantive assertions had already passed; finetune-identity closed
19/19 with the four anchor legs green in the same pass.

**Delta D-TPE2-3 (2026-07-10, Architect):** the predictor lifecycle leg
demanded bitwise across DIFFERENT compute paths (batch-1 dict-input
inference vs a row sliced from the batch-64 reference) and failed at
exactly one float32 ulp (1.19e-07) after TPE-2's drifted-when-present
branch shifted the kernel path. Third instance of the kernel-reassociation
caveat (pre-authorized in the FTW audit): bitwise stays demanded ONLY on
same-computation legs (save->rebuild->composed-predict, 0.00e+00 this
run; the refined-base leg, 0.00e+00); the predictor leg relaxes to 1e-6.
41/42 substantive assertions were already green, including both refined
conditions (drifted-base bitwise + two-way consistency).
