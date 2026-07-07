---
name: cobaya-theory-adapter
description: "SPEC 2026-07-07 (Architect): the Cobaya Theory block for MCMC inference — cobaya_theory/emul_cosmic_shear.py (folder + class name fixed by user; the likelihood contract get_cosmic_shear/state['cosmic_shear'] preserved). Three-layer design: (1) EmulatorPredictor in the PACKAGE (new emulator/inference.py) — wraps rebuild_emulator and owns ALL prediction physics: encode via the saved ParamGeometry, model forward, the factored-IA closed-form combine (nla/tatt coeffs at inference), the NPCE base recombine (residual/ratio), decode via the saved DataVectorGeometry; (2) the Theory file = a THIN cobaya adapter (requirements from the h5's stored parameter names, calculate -> predictor, device pick) — defines NO nn.Module, duplicates NOTHING; (3) the MCMC YAML shrinks to {device, emulators: ['<path root>']} — the legacy ord/extrapar/duplicated-emulator.py all DIE (each was a drift channel; the h5 schema-v2 recipe + stored names replace them, the never-trust-defaults rule applied to the sampling YAML). v1 files refused. Sequenced AFTER save-schema-v2 (needs rebuild_emulator + model_recipe + stored names). Gates GCT-A..C incl. the training-vs-inference parity probe. Decision points flagged (kept-vs-full dv shape vs the C likelihood; TPU dropped; compile-at-inference off by default). NOT implemented."
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

## Handoff (HOLD until save-schema-v2 lands)

### ARCHITECT_HANDOFF
Task: the Cobaya Theory adapter (spec:
notes/cobaya-theory-adapter.md in full; the thin-adapter discipline
and the never-trust-defaults rule are binding — no physics in the
adapter, nothing re-declared in the MCMC YAML that the h5 records).
Base: the save-schema-v2 commit; `git log -1` must show it — else
STOP. Scope: emulator/inference.py (EmulatorPredictor with the
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

SPEC DELIVERED 2026-07-07, NOT implemented, HOLD behind
[[save-schema-resolved-config]] (needs rebuild_emulator +
model_recipe + stored names). Queue: NPCE (in flight) ->
save-schema-v2 -> THIS. Suggested commit sentence: "Cobaya Theory
adapter: emulator/inference.py EmulatorPredictor (encode -> forward
-> factored/NPCE combine -> decode, from the h5 alone) +
cobaya_theory/emul_cosmic_shear.py thin adapter — ord / extrapar /
duplicated architectures all retired by the schema-v2 artifact
(gates GCT-A/B Architect-verified)".

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
