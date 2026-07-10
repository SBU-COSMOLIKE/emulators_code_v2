# Scalar parameter emulators (spec)

**Date:** 2026-07-10. **Status:** SPEC (Architect, Fable). **Spec code:**
SPE. **Home note** for the gates `scalar-identity` / `scalar-smoke`.

## The request (user design goal)

Extend the library beyond cosmolike data vectors: emulators whose output is
a SMALL SET OF NAMED SCALARS — derived parameters. The driving cases are the
legacy emultheta ((omegabh2, omegach2, thetastar) -> H0, omegam: the map
that lets cosmolike run while the sampler walks thetastar) and emulrdrag
((omegabh2, omegach2) -> rdrag). The legacy classes (user-provided v1/v2
under Downloads/emulators_code-main) hardcode one getter method per output
(get_H0 / get_omegam / ...) and require a manual `provides:` list in every
YAML plus per-emulator file/extra/ord/extrapar lists. The unifying class
must derive the provides list AUTOMATICALLY from the artifact.

**The unifying principle (never-trust-defaults applied to `provides`):**
a schema-v2 artifact records its input names already; a scalar artifact
also records its OUTPUT names. The cobaya theory class reads both from the
file — requirements and provides are artifact facts, never YAML restated.

## Design rules

### D-SP1 — the scalar output geometry

New `ScalarGeometry` (emulator/geometries_scalar.py): `names` (the output
parameter names, e.g. ["H0", "omegam"]), `center`, `scale` (per-output
standardization from the training targets — persist-resolved-values), with
encode/decode + state()/from_state + the h5 `cls` marker, so save_emulator
/ rebuild_emulator generalize with near-zero new persistence code. No mask,
no Cinv, no probe — those are dv concepts.

### D-SP2 — training: inputs and outputs are named columns of one dump

The scalar training set is the existing param dump machinery, twice over:
INPUTS = the covmat header names (the standing convention, whitened by the
plain ParamGeometry); OUTPUTS = named columns of the same params .txt,
listed in the YAML:

```yaml
data:
  train_params: chain_thetastar_lcdm.1.txt
  train_covmat: chain_thetastar_lcdm.covmat
  val_params:   chain_thetastar_lcdm_val.1.txt
  outputs:
    - H0
    - omegam
  n_train:    100000
  n_val:      20000
  split_seed: 0
train_args:
  # the standard blocks (model / optimizer / lr / scheduler / loss...);
  # small resmlp widths are plenty for scalar maps
```

No dv files, no cosmolike keys, no cosmolike import anywhere on this path.
A new thin driver `train_scalar_emulator.py` (the train_single skeleton
minus the dv/cosmolike legs); the loss is a standardized mean-square error
(`emulator/losses/scalar.py`) exposing the loop's interface (encode /
loss / chi2-as-metric on standardized outputs, and the output-count
attribute the loop sizes the model by). Existing trim/focus/ema/anchor
machinery composes untouched (they act on per-sample losses).

### D-SP3 — the artifact

Standard schema v2: model recipe + ParamGeometry (inputs) + ScalarGeometry
(outputs, with the `cls` marker dispatching from_state) + resolved config +
histories + identity attrs. rebuild_emulator returns the quad with the
scalar geometry in the dv slot and `info["scalar"] = True` (or dispatch on
the geometry class — Implementer's choice, loud either way). Fine-tune
(FTW) composes automatically once the source constraints admit the scalar
geometry; transfer (TPE) is OUT OF SCOPE for scalars V1 (recorded).

### D-SP4 — the cobaya theory block: provides derived from the file

One generic class, `cobaya_theory/emul_scalars.py` (beside the cosmic-shear
adapter, same conventions: python_path, ROOTDIR-relative roots):

```yaml
theory:
  emul_scalars:
    python_path: ./external_modules/code/emulators_code_v2/cobaya_theory/
    extra_args:
      device: 'cuda'
      emulators:
        - projects/lsst_y1/emulators/thetaH0/emul_v2
        - projects/lsst_y1/emulators/rdrag/emul_v2
```

- NO `provides:` key needed: `get_can_provide_params()` returns the UNION
  of the loaded artifacts' output names; a generic `get_param(name)`
  serves any of them from the per-point cache (this removes the legacy
  one-getter-per-output pattern entirely). An optional YAML `provides:`
  is accepted as a CHECK ONLY (must be a subset of the artifact union;
  mismatch is a loud error naming both lists) — never a source.
- `get_requirements()` = the union of the artifacts' stored input names
  (minus any name another loaded artifact provides, so thetaH0 -> H0
  chains into an H0-consuming scalar emulator if ever wanted; V1 may
  simply forbid overlap with a loud error — Implementer proposes, the
  simpler ruling wins).
- Two artifacts providing the SAME output name = loud error at initialize.
- calculate(): one batch-1 encode/forward/decode per artifact per point;
  outputs cached on the state; scalars also exposed as derived.

### D-SP5 — direct scripting use (the v2 pattern)

The existing EmulatorPredictor grows a scalar branch (dispatch on the
rebuilt geometry class): `predict(dict) -> {name: value}` — the profile
script's `etheta.calculate({...})['H0']` becomes
`predictor.predict(row)["H0"]` with zero per-emulator classes.

### D-SP6 — legacy compatibility: replaced, not ported

The GP/joblib and .pt/extra/ord legacy artifacts stay on the legacy
classes (untouched, still usable). The new path produces schema-v2
artifacts by RETRAINING — scalar maps are cheap to retrain, and porting
the joblib format would drag sklearn into the artifact contract. Recorded
as the deliberate trade.

### D-SP7 — gates

- `scalar-identity` (SPE-A; Mac+board, torch only): synthetic columns ->
  train a tiny scalar emulator -> save -> rebuild -> predict round-trip
  bitwise (same-path); auto-provides == stored output names; the subset
  check and the duplicate-output error fire; ScalarGeometry state
  round-trip byte-identical.
- `scalar-smoke` (SPE-B; board): train a DETERMINISTIC derived target
  from the existing dumps (e.g. omegamh2 = omegam*(H0/100)^2 built from
  the dump's own columns), 2 epochs + full n_train sanity: val must
  collapse (a deterministic smooth map), the saved artifact's
  provides/requirements read back correctly, and a cobaya evaluate run
  through emul_scalars returns the emulated value at the test point
  (the cobaya-adapter gate pattern).

### D-SP8 — out of scope (recorded)

Transfer/refine over scalar emulators; GP (joblib) as a first-class MLA;
the legacy emulcmb/emulbaosn spectra emulators (a different unit: their
outputs are vectors with their own geometries); chained scalar emulators
(unless the D-SP4 overlap ruling admits them trivially).

## Links

[[finetune-warm-start]] (schema/geometry conventions), [[gates-harness-user-run]],
[[py-module-style-conventions]], [[docs-plain-language-define-or-drop]].

## Resume state (Implementer appends below)

### SPE increment 1 (2026-07-10, Opus): core geometry + loss landed + Mac-gated

**Landed (uncommitted on claude/amazing-keller-e798b6, base a3c93fb):**
- `emulator/geometries_scalar.py` = `ScalarGeometry` (D-SP1): names / center /
  scale per-output standardization; `encode` = (y-center)/scale, `decode` =
  t*scale+center; `from_targets` (mean + population-std, a zero-variance column
  is a loud error), `from_state`, `state()`. dest_idx = arange(n_out) and
  total_size = n_out are DERIVED from names in __init__ (an identity, not a
  persisted knob), so the loop sizes the model by chi2fn.dest_idx.numel() ==
  n_out with no scalar branch. No cosmolike / mask / Cinv / probe.
- `emulator/losses/scalar.py` = `ScalarChi2(CosmolikeChi2)` (D-SP2): overrides
  ONLY chi2 (= (pred-target)^2 summed over outputs, a diagonal unit-variance
  Mahalanobis); inherits encode/decode (forward to geom), loss/_reduce (the
  shared trim/mode/focal/berhu reduction), dest_idx/total_size properties, and
  needs_params-absent (so False, param-unaware). `make_scalar_chi2(geom)`
  factory mirrors make_chi2.

**Loop-composition proof (why the core needs zero training.py change).** The
loop touches chi2fn only via `dest_idx` (sizing, training.py:2430), `encode`,
`chi2(pred=, target=)`, and `loss(...)` (1429-1439, 1730-1749); it NEVER calls
geom.squeeze / unsqueeze or chi2(full=True). ScalarGeometry supplies
encode/decode/dest_idx/total_size; ScalarChi2 supplies chi2 and inherits the
rest. Persistence is already generic: save writes `dv_geometry` cls =
type(geom).__qualname__ and rebuild dispatches
importlib.import_module(...).from_state (results.py:492-494), so ScalarGeometry
saves/rebuilds with ZERO results.py change for the geometry itself.

**Mac gate (increment 1).** py_compile OK; numpy probe ALL PASS,
decode(encode(y)) round-trip max|dy| = 0.00e+00, standardized mean/std near
0/1, chi2 == sum sq resid shape (B,), dest_idx sizing, zero-variance raises,
state round-trip byte-identical; AST, unique defs + ScalarChi2 overrides only
chi2 + make_scalar_chi2 returns ScalarChi2(geom=geom). torch legs deferred to
the board (the FTW evidence pattern, [[dev-machine-mac-m2-32gb]]).

### D-SP4 proposal (input overlap): the SIMPLER ruling = FORBID overlap (V1)

The handoff pre-authorizes "Implementer proposes, the simpler ruling wins."
PROPOSAL: V1 forbids input/provide overlap with a loud error at initialize.
`get_requirements()` = the plain union of every loaded artifact's stored input
names; `get_can_provide_params()` = the union of the output names. If any name
is BOTH required by one artifact and provided by another (a would-be chain,
e.g. a thetaH0 emulator provides H0 and another artifact requires H0), raise at
initialize naming the offending name and the two artifacts. Rationale: chaining
needs a topological calculate() order + cycle detection + a
provided-minus-required subtraction that only pays off for a use case no
current artifact needs (emultheta / emulrdrag read cosmological params, not
each other's outputs); the loud error is safe (a chain cannot be built by
accident) and D-SP8 already lists chained scalar emulators as out of scope
unless the ruling admits them trivially. A duplicate output name across two
artifacts stays a separate loud error (D-SP4). Recommend ACCEPT; union + forbid
is about ten lines against a scheduler.

### SPE increment 2 (the wiring): plan for audit, not yet built

Files + integration points (read-confirmed except where flagged NEEDS READ):
1. `emulator/results.py` = add `info["scalar"] = isinstance(geom,
   ScalarGeometry)` to the rebuild return dict (603-610), with a local import;
   geom already rebuilds AS ScalarGeometry via the generic dispatch (D-SP3).
2. `emulator/data_staging.py` = a scalar staging path: OUTPUTS are named columns
   of the SAME params .txt (D-SP2), picked by `data.outputs`, staged as the
   "dv" the ScalarGeometry standardizes; INPUTS = the covmat-header params
   (unchanged). NEEDS READ: load_source + the source-dict shape, to add an
   output-column selector with no cosmolike.
3. `emulator/experiment.py` = from_config + build_geometry + DATA_KEYS + train
   scalar branch: pgeom = ParamGeometry.from_covmat(inputs), geom =
   ScalarGeometry.from_targets(output cols), chi2fn = make_scalar_chi2(geom);
   output_dim = geom.dest_idx.numel(); recipe carries ia=None. `data.outputs`
   REQUIRED, cosmolike keys FORBIDDEN on a scalar run (exclusive, loud). NEEDS
   READ: build_geometry body + from_config validation + DATA_KEYS + build_specs
   recipe assembly + a forward-walk of EVERY cfg access on the scalar path (the
   FTW model-block lesson, [[finetune-warm-start]]).
4. `train_scalar_emulator.py` = thin driver, train_single minus the
   dv/cosmolike legs (no probe-derived model name, no build_shear_angle_map, no
   rescale). NEEDS READ: train_single_emulator_cosmic_shear.py skeleton.
5. `emulator/inference.py` = EmulatorPredictor scalar branch (D-SP5): dispatch
   on info["scalar"], predict(dict) -> {name: value} via the scalar decode, no
   unsqueeze / section slice. NEEDS READ: EmulatorPredictor.
6. `cobaya_theory/emul_scalars.py` = generic theory class (D-SP4): provides /
   requirements read from the artifacts; get_can_provide_params union, get_param
   from a per-point cache, calculate() one batch-1 predict per artifact, an
   optional YAML provides: as a subset-check-only, duplicate-output + overlap
   loud errors. NEEDS READ: cobaya_theory/emul_cosmic_shear.py + a legacy
   emultheta / emulrdrag (Downloads/emulators_code-main) for the cobaya API
   surface (initialize / get_requirements / get_can_provide_params / get_param /
   calculate / must_provide).
7. Gates `scalar-identity` (SPE-A) + `scalar-smoke` (SPE-B) + board configs +
   `example_yamls/scalar_emulator_*.yaml` + README (define-or-drop). Gate homes
   mirror gates/checks/finetune_identity.py / transfer_identity.py.

Increment-2 discipline: forward-walk the WHOLE scalar driver path before
writing; standard schema v2 throughout; no cosmolike import on the scalar path;
CME (unit 2) starts only after SPE closes.

**Handoff of record:** the unified SPE + CME ARCHITECT_HANDOFF lives in
[[cmb-spectra-emulators]] (one implementation pass, SPE first).
