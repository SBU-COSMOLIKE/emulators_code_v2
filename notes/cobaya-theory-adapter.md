---
name: cobaya-theory-adapter
description: "SPEC 2026-07-07 (Architect): the Cobaya Theory block for MCMC inference — cobaya_theory/emul_cosmic_shear.py (folder + class name fixed by user; the likelihood contract get_cosmic_shear/state['cosmic_shear'] preserved). Three-layer design: (1) EmulatorPredictor in the PACKAGE (new emulator/inference.py) — wraps rebuild_emulator and owns ALL prediction physics: encode via the saved ParamGeometry, model forward, the factored-IA closed-form combine (nla/tatt coeffs at inference), the NPCE base recombine (residual/ratio), decode via the saved DataVectorGeometry; (2) the Theory file = a THIN cobaya adapter (requirements from the h5's stored parameter names, calculate -> predictor, device pick) — defines NO nn.Module, duplicates NOTHING; (3) the MCMC YAML shrinks to {device, emulators: ['<path root>']} — the legacy ord/extrapar/duplicated-emulator.py all DIE (each was a drift channel; the h5 schema-v2 recipe + stored names replace them, the never-trust-defaults rule applied to the sampling YAML). v1 files refused. Sequenced AFTER save-schema-v2 (needs rebuild_emulator + model_recipe + stored names). Gates GCT-A..C incl. the training-vs-inference parity probe. Decision points flagged (kept-vs-full dv shape vs the C likelihood; TPU dropped; compile-at-inference off by default). IMPLEMENTED 2026-07-07 (Opus, base a5dd04f), uncommitted; GCT-A/B PASS; dv-shape DECLARED = kept entries (matches legacy res[0]); rebuild_emulator extended to a 4-tuple (+ info{ia,pce_base,pce_form} + compile_model flag); GCT-C parity probe rides the cocoa-env queue."
metadata:
  node_type: memory
  type: project
---

# The Cobaya Theory adapter: MCMC inference on the saved emulator

User directive 2026-07-07: after the save unit, a cobaya Theory
block so the trained emulator runs in an MCMC; legacy =
emul_cosmic_shear.py (the Theory, name KEPT) + emulator.py
(duplicated architectures) + the evaluate YAML (the calling
convention); the new file lives under cobaya_theory/.

## What the legacy teaches (and what dies)

The legacy Theory rebuilds the network from `extrapar` hand-typed in
the MCMC YAML (MLA / INT_DIM_RES / ...), orders inputs from a
hand-typed `ord` list, whitens with hand-saved mean/std, and imports
architectures from a COPY of the model code living inside cobaya.
Every one of those is a default/definition-drift channel — the exact
failure family the [[save-schema-resolved-config]] standing rule
("artifacts never trust defaults") exists to kill. DIES: `extrapar`
(the h5 model_recipe replaces it), `ord` (ParamGeometry.state()
stores the names in order), the duplicated emulator.py (the training
package is the single definition; rebuild_emulator instantiates it),
manual mean/std whitening (the saved geometry's encode/decode are
the transform). KEPT: the class/file name emul_cosmic_shear, the
likelihood contract (calculate -> state["cosmic_shear"],
get_cosmic_shear), the extra_args list shape (multi-emulator ready,
imax=1 today), the device pick (cpu / cuda / mps; TPU dropped —
decision point, easy to re-add).

## The three layers

1. EmulatorPredictor (NEW emulator/inference.py — the physics lives
   in the PACKAGE, testable against the training stack):
   - built from rebuild_emulator(path_root, device) (schema v2;
     v1 refused loudly);
   - predict(params_dict | ordered array) -> the physical data
     vector: encode via the saved ParamGeometry (names + center +
     eigenbasis — the real whitening, not mean/std), model forward
     (eval mode, no_grad), THE FACTORED COMBINE when the recipe says
     ia (nla_coeffs / tatt_coeffs applied to the emitted templates —
     training did this in the loss; inference must do it here), THE
     NPCE RECOMBINE when the h5 carries a pce group (residual: base
     + net; ratio: base * (1 + net)), decode via the saved
     DataVectorGeometry (unwhiten + center);
   - exposes .names (the required parameter list, in order) and
     .total_size / .dest_idx for the shape contract. AUTHORITY
     CHAIN (user-confirmed 2026-07-07): the saved ParamGeometry is
     the one source — .names IS the geometry's stored names (for a
     factored emulator the AmplitudeFactorGeometry's names already
     include the IA amplitudes, so they join requirements
     automatically); the Theory asks the predictor, the predictor
     asks the geometry, and NOBODY keeps a second list;
   - batch-1 friendly (MCMC latency); optional torch.compile OFF by
     default (decision point).
2. cobaya_theory/emul_cosmic_shear.py — the THIN adapter (target:
   under ~100 lines, NO nn.Module, NO physics):
   - extra_args: device + emulators: ['<path root>'] (ONE root per
     emulator -> .emul + .h5 pair; list-shaped for future probes,
     imax = 1 now);
   - initialize(): build one EmulatorPredictor per root;
     requirements = the union of predictor.names — the h5 defines
     the ordering, the YAML never re-declares it;
   - calculate(): X from params in predictor.names order ->
     state["cosmic_shear"]; get_cosmic_shear() as legacy.
   - sys.path handling: the folder sits beside emulator/ at the repo root (cocoa clone: external_modules/code/emulators/emultrfv2/);
     the adapter prepends its parent dir so `import emulator`
     resolves (the drivers' precedent).
3. The MCMC YAML (the new calling convention, example shipped as
   cobaya_theory/EXAMPLE_EMUL_EVALUATE.yaml mirroring the legacy
   example):

       theory:
         emul_cosmic_shear:
           path: ./external_modules/code/emulators/emultrfv2/cobaya_theory/
           stop_at_error: True
           extra_args:
             device: 'cuda'
             emulators: ['projects/lsst_y1/emulators/<run>/emul_v2']

   No ord. No extrapar. The file IS the emulator.

## Decision points (flagged for the user / verified at build)

- The dv SHAPE contract with the C likelihood (use_emulator path):
  legacy returned the kept-entry vector (780). The predictor decodes
  to kept entries; whether the likelihood wants kept or
  full-scattered (unsqueeze to total_size) is VERIFIED against the
  likelihood source at build — match the legacy semantics exactly.
- TPU support dropped (torch_xla); re-adding is a device-string
  case.
- torch.compile at inference: off by default (batch-1 MCMC; the
  compile latency rarely pays off) — an extra_args switch.
- The factored emulators require the IA amplitudes among the
  sampled params (they are in .names via the amplitude-factored
  geometry) — the combine consumes them; documented.

## Gates

- GCT-A (Mac, stubbed): predictor logic pure legs — names/order
  extraction; the factored-combine branch (nla polynomial on
  fabricated templates == hand computation); the pce recombine
  branches; v1-file refusal; the adapter's requirements dict from
  names.
- GCT-B (Mac, static): the thin-adapter discipline — cobaya_theory/
  defines no nn.Module, imports architectures from NOWHERE but the
  package (grep); no ord/extrapar keys accepted (an unknown
  extra_args key errors loudly, naming the v2 convention); docs
  (README Run-it gains an "use it in an MCMC" pointer section; the
  code map gains inference.py + cobaya_theory/; the example YAML
  ships).
- GCT-C (workstation / cocoa env, rides the queue): THE PARITY
  PROBE — predictor(params) vs the training-side eval on the same
  probe points (bitwise / rtol 1e-6); the legacy-style EXAMPLE
  evaluate run end-to-end against the lsst_y1 likelihood
  (use_emulator: 1) with the printed datavector compared against
  the training-side prediction; an MCMC evaluate + a short chain
  smoke.

## Handoff (hold RELEASED 2026-07-07: save-v2 = b2afca6 landed,
## generator unit = a5dd04f landed; ready to relay)

### ARCHITECT_HANDOFF
Task: the Cobaya Theory adapter (spec:
notes/cobaya-theory-adapter.md in full; the thin-adapter discipline
and the never-trust-defaults rule are binding — no physics in the
adapter, nothing re-declared in the MCMC YAML that the h5 records).
Base: commit a5dd04f ("Training-set generator imported verbatim
..."); `git log -1` must show it — else STOP. Both prerequisites
are inside it: save-schema-v2 (rebuild_emulator + model_recipe +
stored names, b2afca6) and the generator appendix. Scope: emulator/inference.py (EmulatorPredictor with the
factored + NPCE branches), cobaya_theory/emul_cosmic_shear.py (the
thin adapter, class name kept), the example evaluate YAML, docs.
The dv-shape contract is VERIFIED against the likelihood source and
matched to legacy semantics (declare which). Gates GCT-A/B on the
Mac; GCT-C embedded for the cocoa-env queue. Report:
IMPLEMENTER_HANDOFF + resume state appended here, raw gate outputs,
deviations declared. Do not commit: print the suggested commit
command.
### END

## Status

IMPLEMENTED 2026-07-07 (Opus, base a5dd04f), uncommitted.
Architect audit: ACCEPTED WITH ONE BLOCKING DELTA (D-CT1 below) +
two micro deltas; the delta pass LANDED 2026-07-07 (Opus) and
CLOSED under independent Architect probes (see Delta closure) —
COMMIT-READY, ONE combined commit. The LAST unit of the arc:
generate -> train -> save -> SAMPLE. GCT-C parity probe (now incl.
the factored save->rebuild->predict round-trip) rides the cocoa
queue.
Suggested commit sentence: "Cobaya Theory
adapter: emulator/inference.py EmulatorPredictor (encode -> forward
-> factored/NPCE combine -> decode, from the h5 alone) +
cobaya_theory/emul_cosmic_shear.py thin adapter — ord / extrapar /
duplicated architectures all retired by the schema-v2 artifact;
geometry class persisted in the h5 so factored saves rebuild
(gates GCT-A/B Architect-verified)".

## Architect audit verdict (2026-07-07, Fable, independent probes)

ACCEPTED WITH ONE BLOCKING DELTA + two micro deltas. What I
re-verified from raw evidence:

- The decode-reuse claim is REAL: TemplateFactoredChi2.decode(pred,
  params_whitened) (IA/loss_functions.py:290),
  PCEResidualChi2.decode -> geom.decode(y + base) (:109),
  PCERatioChi2.decode -> b*(1+pred) (:261) — every signature matches
  the predictor's uniform (pred, x_enc) call; constructors match
  (geom/coeff_fn/n_amps; geom/pce). TemplateMLP.forward slices
  x[:, :n_in] itself (IA/emulator_designs.py:117), so passing the
  full encoded vector uniformly is correct. geom.decode returns
  (B, n_keep) kept entries — the kept-entry contract holds on every
  branch. AmplitudeFactorGeometry: encode = [whitened non-amps ;
  raw amps]; .names (full raw order, amps included), .n_amps,
  .pg_keep all exist as the predictor assumes.
- The recipe stores "ia" unconditionally (experiment.py:1743), so
  rebuild's _need(recipe,"ia") is safe. Caller sweep: the ONLY code
  consumer of rebuild_emulator is the predictor — no stale 3-tuple.
- GCT-B re-run: no nn.Module (docstring mention only), the only
  emulator import is EmulatorPredictor; whitelist rejects
  ord/extrapar/extra/file loudly; requirements = names union +
  fast_params passthrough; get_cosmic_shear kept. YAML parses
  (ruby -ryaml; the Mac python lacks pyyaml). Five-rule math
  scanner + anchors: ALL CLEAN on both READMEs; the Run-it MCMC
  paragraph + the code-map inference.py/cobaya_theory/ entries are
  in. py_compile: all three files OK.

### D-CT1 (BLOCKING): rebuild_emulator cannot rebuild a factored
### run's input geometry — persist the geometry CLASS in the h5

The write side is general (save_emulator's write_state recurses;
its own comment names AmplitudeFactorGeometry.pg_keep) and the
save site passes exp.pgeom — a factored (ia: nla/tatt) run SAVES
its AmplitudeFactorGeometry state fine (pg_keep/amp_idx/n_param/
names, nested). But rebuild_emulator hardcodes
ParamGeometry.from_state (results.py:420), which splats
cls(device, **state) — on the factored keys that is a TypeError,
not even a loud named error. The predictor's ENTIRE factored
branch (a headline feature of this spec) is unreachable
end-to-end. GCT-A passed because the stub harness fed a plain
geometry: it verified the branch logic, never the factored
rebuild round-trip.

The output side has the same latent shape, and WORSE failure
class: DiagonalGeometry / BlockDiagonalGeometry share the base
state keys and override only whiten/unwhiten math — a
mis-dispatched rebuild would not crash, it would SILENTLY decode
with the wrong transform. (Neither is config-reachable today —
DataVectorGeometry.from_cosmolike at experiment.py:1487 is the
only constructor — so output-side is future-proofing, input-side
is the live bug.)

FIX (the doctrine applied to class identity — the file must record
WHAT it is, not just its numbers):
- save_emulator: each geometry group (param_geometry, dv_geometry)
  gains a "cls" attr = the geometry's qualified class name,
  materialized at write time from type(obj).__module__ +
  "." + type(obj).__qualname__ — the writing code's own identity,
  never a re-declaration.
- rebuild_emulator: _need the "cls" attr from EACH group,
  importlib-resolve it (the exact pattern the model recipe's cls
  already uses), call THAT class's from_state. A missing marker is
  a loud KeyError naming a re-save — NEVER a silent fallback to
  the base class (the read-side rule). Consequence stated plainly:
  any v2 file saved between b2afca6 and this fix is refused with
  the loud error; schema stays v2 (additive attr), acceptable
  while v2 is days old and no production save exists.
- GCT-A gains: a factored-geometry state round-trip check at
  whatever level the Mac permits (stub write_state/_read_group
  dict-level dispatch if h5py/torch are absent), and the REAL
  factored save->rebuild->predict round-trip is added to GCT-C
  alongside the parity probe.

### D-CT2 (micro, style): the example YAML's active
### `emulators: ['...']` is a flow-style list

House YAML rule: block style. The active key becomes a block
sequence (the commented fast_params example should match for
copyability):

    emulators:
      - projects/lsst_y1/emulators/<run>/emul_v2

### D-CT3 (micro, pre-existing doc bug, fold in since we touch the
### area): PCEResidualChi2.decode's docstring return shape

emulator/PCE/loss_functions.py:117 says "(B, total_size) physical
dv" — the code returns geom.decode(y + base) = (B, n_keep) kept
entries (geometries_output.py:403). One-line docstring fix; born
in the NPCE unit, load-bearing now that the predictor documents
the kept-entry contract.

### GCT-C rider (flag, not a delta): MPS float64

The geometry whitening tensors are float64-heritage (covmat
np.loadtxt); MPS has no float64 — rebuild's .to(device) and the
predictor's _dtype row on device='mps' may fail or need an
explicit downcast. Verify on the dev Mac when GCT-C-dev runs, or
document the adapter's device pick as cuda/cpu for inference.

### Delta closure (2026-07-07, Architect, independent probes)

ALL THREE DELTAS VERIFIED — the unit is COMMIT-READY.
- D-CT1: both geometry groups now persist attrs["cls"] materialized
  from type() at write time (save side read in the diff); the two
  hardcoded from_state calls are GONE (grep = zero) and their
  imports removed. My OWN exec-extraction of the nested
  _rebuild_geometry (verbatim source, stub-driven): the marker
  dispatches to the named class's from_state with "cls" popped and
  (device, cleaned_state) passed; a missing marker raises KeyError
  naming the marker + the re-save remedy. The bonus is real:
  LogParamGeometry (geometries_parameter.py:195) is a ParamGeometry
  subclass the old hardcode would have silently rebuilt as the base
  class. The REAL factored save->rebuild->predict round-trip rides
  GCT-C as specified.
- D-CT2: emulators is a block sequence; the commented fast_params
  matches; ruby -ryaml re-parse OK.
- D-CT3: the residual decode docstring now says (B, n_keep) kept
  entries.
- py_compile: results / inference / PCE.loss_functions / adapter
  all OK; five-rule scanner + anchors ALL CLEAN on both READMEs.

ONE combined commit per ## Status. After it lands the
generate -> train -> save -> sample arc is COMPLETE on the Mac
side; the workstation board carries GCT-C (parity probe + the
factored round-trip + the MPS-float64 check) and GSV-C.

### Delta handoff (relayed 2026-07-07, executed — kept for the record)

### ARCHITECT_HANDOFF
Task: D-CT1 + D-CT2 + D-CT3 (spec: the audit verdict in
notes/cobaya-theory-adapter.md — the fix designs are binding,
including the loud-missing-marker rule and covering BOTH geometry
groups). Base: the uncommitted cobaya-adapter tree on a5dd04f (your
own diffs; verify git status shows them — else STOP). Scope: the
three deltas exactly; no other files. Gates: GCT-A extended per
D-CT1 (dispatch legs: plain rebuilds plain, factored state
dict -> AmplitudeFactorGeometry.from_state, missing "cls" attr =
loud KeyError naming a re-save; the ruby/py YAML re-parse; the
five-rule scanner if any doc line moves). Report:
IMPLEMENTER_HANDOFF + resume state appended here, raw gate
outputs. Do not commit: after this lands the unit gets ONE
combined commit (sentence in ## Status).
### END

## Implementer resume state (2026-07-07, Opus, base a5dd04f) — DONE

Footprint: NEW emulator/inference.py (EmulatorPredictor), NEW
cobaya_theory/emul_cosmic_shear.py (thin adapter) +
cobaya_theory/EXAMPLE_EMUL_EVALUATE.yaml; emulator/results.py
(rebuild_emulator extended); README.md + emulator/README.md (docs).

- LAYER 1 (emulator/inference.py): EmulatorPredictor. predict() =
  pgeom.encode(theta) -> model(x_enc) [eval, no_grad] ->
  _decode(pred, x_enc) -> dv[0].cpu().numpy(). The forward path was
  read from the training stack, not guessed: the factored model
  slices its own amplitude columns (TemplateMLP.forward:
  x[:, :n_in], n_in = input_dim - n_amps), so predict passes the
  FULL encoded vector uniformly for plain / factored. The decoder is
  the EXACT training chi2fn.decode, reused not re-derived:
  factored -> TemplateFactoredChi2(geom, IA_DESIGNS[ia]["coeff_fn"],
  pgeom.n_amps).decode; NPCE residual -> PCEResidualChi2(geom,
  base).decode = geom.decode(y + base); NPCE ratio ->
  PCERatioChi2(geom, base).decode = base_phys * (1 + pred); plain ->
  geom.decode. .names = pgeom.names (authority chain: the geometry
  is the one source; a factored geometry's names already carry the
  IA amplitudes). Exclusivity guard (ia + pce) + unknown-ia/form
  raises.
- LAYER 2 (cobaya_theory/emul_cosmic_shear.py): the thin adapter,
  no nn.Module, no physics. extra_args whitelist {device, emulators,
  fast_params, compile}; an unknown key errors loudly naming the
  retired ord/extrapar/extra/file. requirements = union of
  predictor.names (+ fast_params passthrough). calculate ->
  state["cosmic_shear"]; get_cosmic_shear kept. sys.path prepends the
  repo root (parent of cobaya_theory/) so `import emulator` resolves.
  Device pick cpu/cuda/mps (TPU dropped). Path roots ROOTDIR-relative
  unless absolute.
- LAYER 3 (EXAMPLE_EMUL_EVALUATE.yaml): modernized from the real
  lsst_y1 EXAMPLE_EMUL_EVALUATE1.yaml — the likelihood
  (lsst_y1.cosmic_shear, use_emulator: 1) + the lambda bridge
  (omegabh2/omegach2/logA the user's own) KEPT; the theory block
  shrinks to path + device + emulators: ['<root>']; a commented
  fast_params + compile ship; ord/extrapar/file/extra gone.
- INTERFACE CHANGE (declared, sanctioned by the save-schema forward
  note): rebuild_emulator(path_root, device, compile_model=True) now
  returns a 4-TUPLE (model, pgeom, geom, info) where info =
  {ia, pce_base, pce_form}; the pce form is read from the pce group
  (loud if missing). Verified the only caller is the predictor (no
  stale 3-tuple unpack anywhere).

DECISIONS / DEVIATIONS declared:
- dv-shape contract = KEPT ENTRIES (predict returns geom.decode(...)
  [0], the kept-entry vector, matching the legacy res[0]); NOT
  full-scattered. Verified against the legacy adapter's
  predict_data_vector; GCT-C confirms end-to-end vs the C likelihood
  (use_emulator) on the workstation.
- rescale (RescaledChi2 / ResidualBaseChi2) is OUT of the predictor
  scope, as spec'd (only factored + NPCE branches). A rescale run's
  analytic R is not in the h5 (needs cosmolike at inference), so it
  is inherently outside h5-only reconstruction — a documented
  limitation, not a gap.
- fast_params v1 = requirements passthrough only (per the addendum);
  the in-theory shear-M application (Roman-style) stays a flagged
  GCT-C decision point (verified against the likelihood's
  use_emulator path there).
- multi-emulator: imax=1 today; the adapter accepts the list and
  concatenates dvs in emulators order (reduces to the single vector).
- compile OFF by default in the predictor (batch-1 MCMC); rebuild's
  compile_model flag lets a caller re-enable it.

GATES: GCT-A (numpy-stubbed torch: nla/tatt coeffs == hand, the
combine einsum, the residual/ratio recombine arithmetic, _as_row
ordering + missing/length raises; static: the decoder branch
selection + guards, the 4-tuple + v1 refusal + pce-form read) PASS;
GCT-B (static: no nn.Module, no architecture imports, the extra_args
whitelist, _check_extra_args rejects ord/extrapar/extra/file, the
docs + example YAML, the five-rule math scanner) PASS; py_compile
clean. GCT-C (the parity probe: predictor vs training-side eval,
rtol 1e-6; the legacy-style evaluate end-to-end vs the lsst_y1
likelihood; an MCMC smoke) rides the cocoa-env queue.

## Addendum (2026-07-07): ord's replacement made explicit +
## fast_params added (user catch)

User asked after ord + fast_params (the Roman legacy shape).

ORD stays retired — the deliberate design: the h5 stores the names
in training order (ParamGeometry.state()); requirements + input
ordering come from the artifact (a hand-typed ord that disagrees
with the file = silent garbage, the drift channel the standing rule
kills). The spec + example YAML now say the MAPPING story out loud:
when the sampled parameters differ from the training names, the
cobaya params block bridges with derived lambdas (the user's own
legacy example: omegabh2 / omegach2 / logA lambdas) and the Theory
keeps cobaya's standard `renames` passthrough. The example evaluate
YAML ships WITH such a lambda block so the pattern is copyable.

FAST_PARAMS added (a real gap): optional extra_args key,
list-shaped per emulator like the legacy —

    fast_params: [['roman_M1', ..., 'roman_M8']]

Semantics v1: REQUIREMENTS PASSTHROUGH — the names join
get_requirements (the sampler provides + blocks them as fast, since
this theory is cheap) but do NOT enter the network input vector.
DECISION POINT (verified against the likelihood's use_emulator path
at build): whether the shear-calibration M application lives in the
likelihood (the lsst_y1 pattern — then passthrough is all the
theory ever does with them) or must be applied in-theory to the
emulated dv (xi_ij scaled by (1+m_i)(1+m_j)) for the Roman-style
setup; if the latter, it enters EmulatorPredictor as an optional
analytic post-step (never the network), spec'd then. GCT-A gains
the fast_params cases (joined requirements; excluded from the input
vector; absent = byte-identical). GCT-B: the example YAML carries
both the lambda block and a commented fast_params.

## Implementer D-CT delta pass (2026-07-07, Opus, base a5dd04f) — DONE

All three deltas landed; the combined cobaya-adapter unit is
COMMIT-READY. Files touched: emulator/results.py (D-CT1),
cobaya_theory/EXAMPLE_EMUL_EVALUATE.yaml (D-CT2),
emulator/PCE/loss_functions.py (D-CT3).

- D-CT1 (the blocking bug): the geometry CLASS is now persisted and
  dispatched, both groups. save_emulator writes each group's "cls"
  attr materialized from type(obj).__module__ + "." +
  type(obj).__qualname__ (param_geometry from type(param_geometry),
  dv_geometry from type(geometry)). rebuild_emulator gained a nested
  _rebuild_geometry(group, where): _read_group -> require "cls"
  (loud KeyError NAMING a re-save if absent, never a base-class
  fallback) -> pop it -> importlib-resolve (the model-recipe
  pattern) -> THAT class's from_state(device, cleaned_state). The
  two hardcoded ParamGeometry/DataVectorGeometry.from_state calls +
  their now-unused imports are gone. This also fixes a latent
  LogParamGeometry mis-rebuild (the old hardcode would have decoded a
  log run with linear whitening) and future-proofs the Diagonal /
  BlockDiagonal output geometries (shared base keys, different
  transform -> a silent wrong decode without the marker). Schema
  stays v2 (additive attr); any v2 file saved between b2afca6 and
  this fix is refused with the loud re-save error (acceptable: v2 is
  days old, no production save exists). The h5-layout docstring notes
  the "cls" attr on both groups.
- D-CT2: the example YAML's active `emulators` key is now a block
  sequence (- projects/...); the commented fast_params example
  matches block style for copyability.
- D-CT3: PCEResidualChi2.decode's docstring return shape fixed to
  (B, n_keep) kept entries (was the stale "(B, total_size)"; the
  code returns geom.decode(y + base) = kept entries).

GATES: GCT-A extended and re-run — the dispatch legs verified by
exec-extracting the nested _rebuild_geometry and driving it with
stubs (marker class resolved, "cls" popped before from_state, a
missing marker raises a loud KeyError naming a re-save), plus the
static save/rebuild/import checks; D-CT2 verified block-style +
re-parsed with ruby -ryaml; D-CT3 docstring checked. ALL GCT-A/B +
D-CT1/2/3 PASS; five-rule math scanner CLEAN (no README line moved);
py_compile clean (results, PCE/loss_functions, inference, adapter).
The REAL factored save->rebuild->predict round-trip is added to
GCT-C (workstation) alongside the parity probe.

FLAG carried (not a delta, Architect's GCT-C rider): MPS has no
float64; the geometry whitening tensors are float64-heritage, so on
device='mps' rebuild's .to(device) / the predictor's _dtype row may
need an explicit downcast or the adapter documents cuda/cpu for
inference — verified when GCT-C-dev runs.

### 2026-07-08 — Architect: cobaya loads the class via python_path, not path

Board run 5 was the first time cobaya-run actually resolved the theory
block, and it exposed a wiring error in both evaluate YAMLs: the key that
points cobaya at an external class is `python_path` (cobaya/input.py:318
reads exactly that key into component_path); `path` is a different
component attribute (the install path of driven code, camb-style) and is
ignored for class lookup. Without python_path, cobaya's internal lookup
found the LEGACY v1 adapter that cocoa's cobaya fork bundles under the
same name (cobaya/theories/emul_cosmic_shear/ — it died at initialize
with "Missing emulator file (extra) option", the retired file/extra
split-pair convention). Keeping the class name emul_cosmic_shear is
still right (the likelihood contract expects it); python_path makes the
external class win or fail loudly (cobaya/component.py:650: "If
component_path is specified, load the class from there or fail") — the
builtin can never silently shadow it again. Both YAMLs (the board's
gates/configs/cobaya-adapter-evaluate.yaml and the shipped
EXAMPLE_EMUL_EVALUATE.yaml) now use python_path with a comment naming
the trap.

## Addendum (2026-07-08): dv-shape decision point RESOLVED — section-sized products, a shape flag, sections persisted at save

Board run 8 executed the adapter inside cobaya for the first time and
falsified the implementation-time declaration "dv shape = kept entries
(matches legacy res[0])". The likelihood contract
(_cosmolike_prototype_base.internal_get_datavector_emulator, probe xi)
is len(get_cosmic_shear()) == sizes[0] = the FULL cosmic-shear section
as cosmolike sizes it (780 for lsst_y1) — the legacy v1 adapter emitted
exactly that (OUTPUT_DIM 780). Kept entries equal sizes[0] only when
the mask keeps the whole section; with lsst_y1_M1_GGL0.05 it does not.

**USER DESIGN DECISION (overrides the Architect's first proposal of a
full-length product + a likelihood check relaxation):** the likelihood
stays UNTOUCHED — it is the component that glues per-probe products
(cosmic_shear / ggl / wtheta, each section-sized) into the full 3x2pt
vector, and separate per-probe emulators are the intended future. So:

1. predict() gains a SHAPE FLAG: `dv_return: 'section' | '3x2pt'`,
   DEFAULT 'section'. Section mode returns the emulator's own probe
   section(s) sliced from the scattered full vector — for a cosmic-shear
   emulator that is exactly the xi± block, full[0:sizes[0]], the length
   the likelihood demands. '3x2pt' returns the full-length scattered
   vector (masked positions zero). The flag chooses SHAPE only; WHICH
   section comes from the artifact (never re-declared in the YAML).
   The training-side data vector (dumps, mask, center, Cinv) always
   stays full-3x2pt-length — completely separate story, unchanged.

2. The artifact must therefore record the section boundaries:
   DataVectorGeometry.from_cosmolike ALREADY resolves
   ci.compute_data_vector_3x2pt_real_sizes() and the probe string at
   staging (geometries_output.py:230-247) but state() drops them.
   state()/from_state gain `section_sizes` (the 3-list) and `probe`
   (the possible_probes string) — training-time RESOLVED values, the
   persist-resolved-values rule applied. An older v2 h5 without the
   keys loads with them as None; section mode then fails LOUDLY naming
   the fix (re-save with current code, or set dv_return: '3x2pt').
   The board's persisted tiny emulator regenerates every run, so the
   gate picks the new keys up automatically.

3. Mechanics of section mode: decode to kept entries exactly as today
   -> scatter via the geometry's own unsqueeze (geometries_output.py:325,
   dest_idx placement into a total_size zero vector) -> slice the stored
   probe's block(s) (block k starts at sum(section_sizes[:k]), length
   section_sizes[k]; multi-block probes concatenate their blocks). For
   probe xi that is full[0:section_sizes[0]] — offset 0, so kept
   positions are unchanged and masked positions are zero, matching the
   legacy 780-vector semantics bit for bit where it matters (compute_logp
   masks them anyway).

4. The adapter's extra_args whitelist gains optional `dv_return`
   (default 'section'), passed through to every EmulatorPredictor. The
   board evaluate YAML stays unchanged (it exercises the default path);
   the shipped example documents the flag commented-out.

The Architect's rejected alternative (full-length product + relaxing
the likelihood's len check) is recorded for the archive: it would have
made the likelihood accept two shapes per product and put the gluing
convention on both sides of the interface; the user's design keeps one
shape per product and one gluer.

### 2026-07-08 — board verdict (Architect): GCT-C cobaya-adapter PASS — the full spec (:117-123) closed
All three named legs are green. (1) Parity: predictor vs training side
rtol 1e-6 (plain worst 3.2e-7, factored 7.9e-7) plus the dv_return
shape assertions measured on hardware (section 780 == section_sizes[0],
3x2pt 1560 == total_size, masked positions exactly 0.0). (2) Evaluate:
rc 0 through the UNTOUCHED likelihood (log-posterior -2490.88; the
chi2 2990.18 is meaningless by design — 3-epoch tiny emulator).
(3) MCMC: 500 accepted steps rc 0; cobaya measured speeds
(emul_cosmic_shear 776 evals/s warm, 1.2 ms/call vs the 0.11 s
compile-inclusive first call) and the fast/slow blocking demonstrably
worked — 1577 likelihood evaluations against 845 theory calls, the
sampler oversampling the LSST_M* fast block on the cached emulator
product. Ten wiring layers peeled across runs 3-10, zero physics bugs;
re-proven post-refactor in run 11.
