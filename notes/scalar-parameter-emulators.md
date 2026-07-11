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

## Architect audit: increment-1 checkpoint (2026-07-10, Fable)

**Verdict: core VERIFIED, with one delta (D-SPE1-1) to fold into
increment 2. D-SP4 RULED (accepted as proposed). Increment-2 plan
APPROVED.** Evidence below is the Architect's own run (independent
exec-probe + AST + mechanical claim checks on the worktree source; the
FTW/TPE probe pattern), not the Implementer's gate output.

### Evidence

- Shipped-span exec probe (numpy torch-shim, 9000x3 targets):
  from_targets center == numpy mean and scale == population std (ddof 0)
  bitwise after the float32 cast; decode(encode(y)) max|dy| = 0.00e+00;
  standardized mean/std within 3.4e-06 / 1.2e-06 of 0/1; dest_idx =
  arange(n_out), total_size = n_out, numel == len(names); state() keys
  exactly {names, center, scale}; from_state(state()) rebuild encodes
  bitwise; make_scalar_chi2 returns a ScalarChi2 holding the geom; chi2
  == sum_i (pred-target)^2 bitwise, keyword call, shape (B,),
  non-negative, exactly 0 at pred == target; full= accepted and inert.
- AST: ScalarChi2's body defines only chi2 (no class attrs, so
  needs_params stays absent); base list == [CosmolikeChi2]; the two
  files import exactly {numpy, torch, .core} — cosmolike-free.
- Loop claims re-verified against the worktree source, not the handoff:
  CosmolikeChi2.__init__ stores geom only (core.py:121-129); dest_idx /
  total_size are forwarding properties (137-144); the inherited loss
  reaches the geometry only through self.chi2(pred=, target=) and
  _reduce is pure reduction; needs_params = True exists only on
  RescaledChi2 / ia / pce / transfer, never the base, so the loop's
  getattr default False holds (training.py:1337/1428/1436/1682); the
  loss call-site kwargs (1730-1738: mode/trim/focus/focus_scale/
  berhu_knot/berhu_cap/berhu_s) all land in the inherited signature;
  out_dim = chi2fn.dest_idx.numel() (2430); no chi2(full=True) anywhere
  (the lone "full =" grep hit is n_full at 1914); no chi2fn.geom.*
  access in the loop.
- Persistence: save writes cls = __module__ + "." + __qualname__
  (results.py:279-280) and rebuild rpartitions on "." (492-494); the
  handoff's "cls = __qualname__" was shorthand, the mechanism is
  correct and the zero-results.py-change claim holds. write_state
  already handles str lists (ParamGeometry's names), so
  ScalarGeometry.state() round-trips with no writer change.

### D-SPE1-1 (delta, REQUIRED in increment 2): the zero-variance guard
is absolute and misses real constants

A constant column at a non-representable value — 0.31, exactly the
omegam-like case — has float64 std 5.55e-17 from mean-rounding, so
`scale <= 0.0` passes and the docstring's promised loud error becomes a
silent scale of 5.6e-17. (The increment-1 probe's constant happened to
be exactly representable, where the std is exactly 0 — the test passed
while the guard does not generalize.) The float32 pipeline adds a
second failure the same guard should own: a relative spread below
float32 resolution at the column's magnitude cannot survive the
float32 cast of center/scale (the cast's rounding, ~6e-8 x |mean|,
swamps the spread). One relative guard catches both; replace the two
guard lines in from_targets:

```python
    # a constant column's std is pure mean-rounding noise
    # (~eps64 * |mean|), and a spread below float32 resolution at the
    # column's magnitude cannot survive the float32 cast; both are
    # un-standardizable, so both are the loud error.
    tiny = 8.0 * np.finfo("float32").eps * np.abs(center)
    zero = np.nonzero(scale <= tiny)[0]
```

Boundary checks that fix the constant: an exact-zero column has
center = 0, tiny = 0, scale = 0, still caught by <=; a legitimate
tiny-magnitude output (center 1e-9, 10% spread: std 1e-10) passes its
threshold of ~1e-15 with ten orders to spare. Extend the error message
to name both failure modes (constant column, or spread below float32
resolution). Two regression legs ride the increment-2 Mac probe AND
the scalar-identity gate: a 0.31 constant column must raise; a
center 1e-9 / 10% spread column must not.

### RULING — D-SP4 accepted as proposed (forbid overlap, V1)

- get_requirements() = the plain union of every loaded artifact's
  stored input names; get_can_provide_params() = the union of the
  stored output names. No subtraction, no chaining, no scheduler.
- At initialize, two separate loud errors: (a) a duplicate output name
  across artifacts, naming the name and both artifact roots; (b) any
  name both required by one artifact and provided by another (the
  would-be chain), naming the name and the two artifacts.
- The D-SP4 parenthetical "minus any name another loaded artifact
  provides" is SUPERSEDED by this ruling; D-SP8's chained-scalar-
  emulators stays out of scope. The YAML provides: subset-check-only
  rule is unchanged. The scalar-identity gate asserts both error paths.

### Increment-2 plan: APPROVED as written

The seven integration points, the NEEDS READ list, and the
whole-driver-path forward-walk discipline stand as specced. One
addition: carry D-SPE1-1 and its two regression legs.

### ARCHITECT_HANDOFF: SPE INCREMENT-1 VERIFIED — PROCEED TO INCREMENT 2

- **Audit outcome:** increment-1 core (geometries_scalar.py,
  losses/scalar.py) VERIFIED against the raw worktree source; every
  loop-composition and persistence claim independently confirmed; the
  full evidence record is the audit section above this block.
- **Delta D-SPE1-1 (required):** replace from_targets' absolute
  zero-variance guard with the relative guard quoted above (8 x
  float32-eps x |center|), extend the error message to both failure
  modes, and add the two regression legs (0.31 constant raises;
  center 1e-9 / 10% spread does not) to the increment-2 Mac probe and
  the scalar-identity gate.
- **RULING D-SP4:** ACCEPTED as proposed — forbid input/provide
  overlap, loud at initialize; duplicate outputs a separate loud
  error; requirements/provides are plain unions; the spec's chaining
  parenthetical is superseded; provides: stays subset-check-only.
- **Increment 2:** the plan in the resume is APPROVED as written; the
  discipline lines (whole-path forward-walk, schema v2, no cosmolike
  import, CME only after SPE closes) are binding.
- **Next milestone:** the full SPE IMPLEMENTER_HANDOFF — increment 2
  built and Mac-gated (probe + AST + compileall), D-SPE1-1 closed,
  gates scalar-identity / scalar-smoke authored with board configs +
  example YAML + README section, and the self-contained force-rerun
  list for the workstation run (the user is away; the handback must
  stand alone).

## Resume: SPE increment 2 in progress (2026-07-10, Opus)

**Base:** claude/amazing-keller-e798b6 @ 5db2a3e (increment-1 commit).

**Landed so far (uncommitted on top of 5db2a3e):**
- D-SPE1-1 CLOSED: `from_targets` now uses the relative guard `tiny =
  8 * float32-eps * |center|; scale <= tiny`, error message names both
  failure modes; docstring updated. Probe extended with the two
  regression legs (0.31 constant RAISES; center 1e-9 / 10% spread does
  NOT) + the source cross-check swapped to the relative guard. Re-run
  ALL PASS.
- D-SP3 (results.py): rebuild return dict gains `"scalar":
  isinstance(geom, ScalarGeometry)` (local import added beside
  make_activation/make_norm). Dispatches on the rebuilt class, older
  artifacts report False.

**Branch divergence (integration note for the user).** Since the fork
point 3e511af, origin/main (38c1442) changed ONLY README.md (the
Architect's +863/-211 rewrite) and does NOT carry the SPE files (the
earlier merge 6edfa2d took a pre-SPE branch state). My branch has SPE
increment-1. So the two diverged on README.md ALONE; every increment-2
CODE surface (results / data_staging / experiment / inference / driver /
cobaya / gates / YAML) is conflict-free. Plan: build all code on this
branch; deliver the README scalar section as a ready-to-apply draft
(written against origin/main's rewritten README) rather than editing the
stale README.md here (which would conflict + misattribute the rewrite).
Recommended integration order for the user: `git merge origin/main` into
this branch (brings the new README in cleanly, code is disjoint), then
apply the README scalar section, then merge to main.

**Still to build (increment-2 plan, all APPROVED):** data_staging output
columns; experiment.py scalar branch + whole-path forward-walk; the thin
driver; EmulatorPredictor scalar branch; cobaya_theory/emul_scalars.py
(D-SP4 forbid-overlap, ruled); gates scalar-identity (+ the two D-SPE1-1
legs) / scalar-smoke + board configs + example YAML + README draft.

### Concrete increment-2 design (from the code reads, ready to implement)

**Scalar staging = `load_scalar_source` in data_staging.py (new).** Both
inputs and outputs are columns of the ONE params .txt (D-SP2), so the
loader reads the .txt once and selects columns by name via the getdist
`.paramnames` sidecar. Mapping: a getdist .txt is `weight, minuslogpost,
<params...>, ...`, so `.paramnames` line i names .txt column `2 + i`
(the existing load_source hardcodes `param_cols=slice(2, -1)` and assumes
those == covmat names, which holds only when the .txt has no derived
columns; a scalar chain HAS derived columns, e.g. H0 / omegam, so it must
select by name). Design: `.paramnames` is REQUIRED for a scalar run (the
only way to locate outputs; absent = loud error). Build
`full_names = [line.split()[0].rstrip("*") for line in .paramnames]`;
`in_cols = [2 + full_names.index(n) for n in self.names]` (covmat order),
`out_cols = [2 + full_names.index(n) for n in self.outputs]`; loud if any
name missing. C = txt[:, in_cols], Y = txt[:, out_cols]. Reuse
phys_cut_idx (on C, names=self.names), stage_source, param_stats for
C_mean. Return `{C, dv: Y, idx, C_mean}` (dv slot = the output targets;
ScalarGeometry.from_targets recomputes center/scale from the staged Y, so
dv_mean is not needed). No cosmolike, no .npy.

**experiment.py branch points (forward-walk done):**
- `__init__`: add `self.outputs` (list) + `self._scalar` (bool) fields.
- `from_config` (~1223): a scalar run = `data.outputs` present; assert
  cosmolike keys (`cosmolike_data_dir` / `cosmolike_dataset`) ABSENT
  (both present = loud, exclusive) and no `train_dv` / `val_dv`; set
  `self._scalar=True`, `self.outputs=data["outputs"]`; forbid ia / pce /
  rescale / transfer / finetune on a scalar run (loud).
- `DATA_KEYS` (355): `outputs` becomes a valid key; the cosmolike keys +
  dv keys are not required on a scalar run (they are required on a dv
  run). Validate exclusivity in the DATA_KEYS check.
- `stage_train` / `stage_val` (1628 / 1670): branch to
  `load_scalar_source(params_path=..., in_names=self.names,
  out_names=self.outputs, <cuts>, ...)` when `self._scalar`.
- `build_geometry` (1748): scalar branch AFTER the transfer branch (1844)
  and BEFORE the cosmolike import (1855): `self.pgeom =
  ParamGeometry.from_covmat(center=C_mean, covmat_path=train_covmat)`;
  `self.geom = ScalarGeometry.from_targets(device, Y[idx], names=
  self.outputs)`; `self.chi2fn = make_scalar_chi2(self.geom)`; early
  return. No cosmolike import reached.
- `build_specs` (1960): output_dim already = `self.geom.dest_idx.numel()`
  == n_out (works unchanged); the recipe must carry `ia=None` and
  `needs_geom=False` for the scalar model (a plain resmlp). Confirm the
  recipe-assembly path sets ia from `self.ia` (None on scalar).
- `print_design` (1468): a scalar banner (outputs list, n_out, no probe /
  cosmolike / rescale lines).

**Model reuse:** a scalar run is `MODELS[(resmlp, None)]` = the plain
ResMLP; no new model class. `self.arch = resmlp`, `self.ia = None`.

**Driver `train_scalar_emulator.py`:** train_single minus `--rescale` /
`--diagnostic` / cosmolike; keep `--activation` / `--quiet`; run_tag =
`<model>_ntrain<N>` (no `_cs_<T>` temperature tag; scalar dumps have no
such tag). save_emulator unchanged (pgeom + ScalarGeometry geom + config
+ histories + resolved recipes; ia/pce/transfer args all None).

**Inference (D-SP5):** EmulatorPredictor branch on `info["scalar"]`:
`predict(dict) -> {name: value}` = encode the input row through pgeom,
forward, `geom.decode` (destandardize), zip with `geom.names`. No
unsqueeze / section slice.

**Cobaya `emul_scalars.py` (D-SP4, ruled):** STILL NEEDS READ of
cobaya_theory/emul_cosmic_shear.py + a legacy emultheta/emulrdrag for the
API surface before writing.

**Checkpoint:** stopped here at a coherent boundary (D-SPE1-1 + D-SP3
landed and gated; the rest fully designed) to avoid a mid-write
compaction on the interconnected experiment.py wiring. Next turn writes
load_scalar_source + the experiment.py branch (all points above), Mac-
gates, then the driver / inference / cobaya / gates / YAML / README.

### Update 2 (2026-07-10, Opus): data_staging landed + gated (D-SPE2-1)

`data_staging.py` gains `_scalar_columns` (pure) + `load_scalar_source`.
- `_scalar_columns(sidecar, in_names, out_names)`: reads the .paramnames
  sidecar, strips the derived `*` marker, D-SPE2-1(a) asserts the name
  list is UNIQUE before any `.index()` (loud, names duplicates), maps
  each name to `.txt` column `2 + line_index`, loud on a missing name.
- `load_scalar_source`: sidecar REQUIRED (loud if absent), runs
  `check_paramnames` (D-SPE2-1b, pins the sampled block to the covmat
  order), slices C = in-cols, Y = out-cols from one `np.loadtxt`, seeded
  shuffle, optional cuts, `stage_source`, returns `{C, dv:Y, idx}` (+
  C_mean / dv_mean with with_means).
- **FORWARD-WALK FINDING (refinement to the plan):** physical param_cuts
  are OPTIONAL on the scalar path (`omegabh2_hi=None` skips phys_cut_idx
  entirely). The omega-windows reference omegab / H0 / omegam / ns, which
  a scalar INPUT set like (omegabh2, omegach2, thetastar) may not carry;
  forcing them would wrongly require those columns. So a scalar chain
  (already the target distribution) runs cut-free by default; cuts are
  opt-in and then loudly require their columns. experiment.py from_config
  must therefore make data.param_cuts OPTIONAL on a scalar run.
- Gate (Mac): py_compile OK; staging probe ALL PASS (exec-extracted
  `_scalar_columns`: mapping [2,3,4]/[5,6], out-order honored, duplicate
  raises, missing input/output raise; source cross-checks incl.
  check_paramnames + cuts-optional). torch legs (randperm shuffle,
  stage_source, param_stats) ride the board.

**Next:** the experiment.py scalar branch (from_config incl. param_cuts
optional, DATA_KEYS, __init__ fields, stage_train/val -> load_scalar_
source, build_geometry scalar branch, build_specs ia=None, print_design),
with the whole-path forward-walk record; then driver / inference /
emul_scalars / gates / YAML / README.

### Update 3 (2026-07-10, Opus): experiment.py scalar branch landed + gated

All eight edit points in experiment.py, additive + scalar-guarded:
import load_scalar_source; DATA_KEYS += "outputs"; new module fn
`validate_scalar` (exclusivity: no dv/cosmolike keys; required param files;
forbids rescale/ia/pce/transfer/finetune; outputs non-empty + unique);
from_config (is_scalar detection, param_cuts made conditional, a scalar
branch building MODELS[(name, None)] with _scalar/outputs set); __init__
(_scalar=False, outputs=None); stage_train + stage_val (scalar branch ->
load_scalar_source before the param_cuts access); build_geometry (scalar
branch after the transfer branch, before the cosmolike import:
ParamGeometry.from_covmat + ScalarGeometry.from_targets(staged targets) +
make_scalar_chi2, early return); print_design (a scalar summary line + the
cuts line gated by `if pc:`, byte-identical for dv since param_cuts is
always non-empty there). build_specs UNCHANGED (recipe["ia"]=self.ia=None,
output_dim=geom.dest_idx.numel()=n_out both already correct).

**Whole-path forward-walk (from_config -> stage -> build_geometry ->
build_specs -> run -> save -> run_tag/diagnostics):** the training path
(run() = stage_train/val -> build_geometry -> train) threads cleanly; save
persists the ScalarGeometry via the generic cls dispatch (no results.py
change beyond the D-SP3 info flag). FINDING: the train_single driver's
`run_tag` reads `cfg["data"]["train_dv"]` (its _cs_<T> tag) and its `attrs`
read `train_dv` / `val_dv` (basenames) -- BOTH absent on a scalar run, so
train_single cannot be reused as-is; the scalar driver MUST supply its own
run_tag (`<model>_ntrain<N>`) and attrs (no train_dv/val_dv). This is
exactly why train_scalar_emulator.py is a separate thin driver, confirmed
by the forward-walk.

**Gate (Mac):** py_compile OK (experiment/data_staging/results/
geometries_scalar/losses.scalar); experiment probe ALL PASS
(validate_scalar valid/empty/dup/exclusivity/required-files/forbidden-
features + 8 source cross-checks); diff 228+/11- (the 11 deletions are the
import reflow + the param_cuts conditional + the cuts-line re-indent under
`if pc:`, all additive/guarded; dv-path load_source x4 / from_cosmolike x1
/ make_chi2 x2 intact). torch training legs ride the board.

**Next:** train_scalar_emulator.py (own run_tag + attrs), EmulatorPredictor
scalar branch, then the binding emul_cosmic_shear + legacy emultheta reads
-> emul_scalars, gates scalar-identity / scalar-smoke + configs + example
YAML + README draft.

### Update 4 (2026-07-10, Opus): D-SPE2-3 + driver + predictor landed + gated

- D-SPE2-3 CLOSED: from_config scalar branch resolves `model_cls =
  models[(name, None)]` and raises if `model_cls.head_block is not None`
  (verbatim from the note); keyed on the declared head_block, not the name,
  so a `name: rescnn` scalar YAML fails loudly at config time instead of
  crashing at build_shear_angle_map. Probe leg added (guard present +
  message + logic shape); the end-to-end rescnn-raises leg rides the
  scalar-identity board gate.
- `train_scalar_emulator.py` (new, thin driver): train_single minus
  rescale / diagnostic / cosmolike; own `run_tag` = `<model>_ntrain<N>` (no
  _cs_<T> temperature tag) and own attrs (train_params / val_params
  basenames + outputs, no train_dv/val_dv, per the forward-walk finding);
  save_emulator with pce=None / transfer_base=None.
- `emulator/inference.py` EmulatorPredictor scalar branch (D-SP5): __init__
  branches on `info["scalar"]` BEFORE the dv-geometry accounting
  (section_sizes / probe, which a ScalarGeometry lacks), sets
  output_names + _dtype, returns early (no _build_decoder); predict()
  destandardizes via `geom.decode(pred)` and returns a `{name: value}`
  dict. dv predict path unchanged (self._decode / unsqueeze intact).

**Gate (Mac):** py_compile OK (experiment / inference / train_scalar_
emulator); experiment probe ALL PASS incl. the new D-SPE2-3 legs; diffs
additive/guarded (experiment +D-SPE2-3; inference 27+/5-, dv path intact).
torch legs (real predict round-trip, save/rebuild, a scalar train) ride the
scalar-identity / scalar-smoke board gates.

**Next:** the BINDING reads (cobaya_theory/emul_cosmic_shear.py + a legacy
emultheta / emulrdrag under Downloads/emulators_code-main) -> write
cobaya_theory/emul_scalars.py (D-SP4 forbid-overlap, ruled); then gates
scalar-identity (bitwise round-trip + auto-provides + subset/dup/overlap +
D-SPE2-3 + D-SPE1-1 legs) / scalar-smoke (deterministic fixture dump +
sidecar per D-SP7) + board configs + example YAML + README draft -> the
full SPE IMPLEMENTER_HANDOFF with the workstation force-rerun list.

### Update 5 (2026-07-10, Opus): cobaya adapter emul_scalars.py landed + gated

Binding reads done: cobaya_theory/emul_cosmic_shear.py (the modern thin
structure: _ALLOWED_EXTRA_ARGS whitelist, _pick_device, initialize builds
one predictor per root, get_requirements) + legacy emultheta.py (the
derived-param mechanism: calculate writes state[name] + state["derived"]
[name], per-getter get_H0/get_omegam that the unified class replaces; it
needed a hand-typed YAML `provides:`). cobaya_theory/emul_scalars.py
written from those sources, NOT the paraphrase:
- get_can_provide_params() = the union of the predictors' output_names
  (artifact-derived, D-SP4); get_param(name) = self.current_state[name],
  the single generic getter replacing the per-output methods.
- get_requirements() = the plain union of the predictors' input names
  (predictor.names). D-SP4 ruled unions, no subtraction.
- initialize: duplicate output name across artifacts -> loud; forbid
  input/provide overlap -> loud (chaining out of scope, D-SP8); optional
  extra_args `provides` = subset-CHECK-only, never a source.
- calculate: one predictor.predict(params) per artifact -> {name: value};
  caches state[name] (for get_param) + state["derived"][name] (when
  want_derived), the legacy's derived mechanism.
- _pick_device / _check_extra_args mirror emul_cosmic_shear (TPU dropped).
- Gate (Mac): py_compile OK; AST/source probe ALL PASS (Theory subclass;
  all 7 methods; the three D-SP4 error branches; get_can_provide returns
  the union; get_param reads the cache; extra_args whitelist; derived
  write). cobaya evaluate + the runtime dup/overlap/subset legs ride the
  scalar-identity / scalar-smoke board gates (cobaya not importable on Mac).

**Next (SPE remainder):** gates gates/checks/scalar_identity.py (bitwise
predict round-trip after save/rebuild; ScalarGeometry state byte-identical;
emul_scalars auto-provides == stored names + the dup/overlap/subset error
legs + D-SPE2-3 rescnn-raises + D-SPE1-1 constant-column legs) and
scalar_smoke (a tiny fixture params .txt + .paramnames with an
exactly-derivable omegamh2 column, 2-epoch train, cobaya evaluate through
emul_scalars) + board registration/configs + example_yamls YAML + the
README scalar-section draft (against origin/main's rewritten README, for
post-sync apply) -> the full SPE IMPLEMENTER_HANDOFF + force-rerun list.

## Architect audit: increment-2 checkpoint (2026-07-10, Fable)

**Verdict: both landed deltas VERIFIED (D-SPE1-1 closed, D-SP3 landed);
the divergence claims verified against git; the staging design is
ENDORSED with one small required addition (D-SPE2-1) and one gate
clarification. Proceed to the wiring.** Evidence is the Architect's own
run on the worktree source.

### Evidence

- D-SPE1-1: the from_targets guard is verbatim to the delta spec
  (tiny = 8 * float32-eps * |center|; scale <= tiny), message and
  docstring name both failure modes. My probe re-run: the previously
  failing 0.31-constant leg now PASSES; my own three regression legs
  all pass — 0.31 constant raises naming the column, center 1e-9 / 10%
  spread builds with standardized std ~ 1, an exact-zero column still
  raises. The increment-1 full probe re-run is green (its one listed
  failure is the audit probe's own blunt "full =" grep, diagnosed at
  increment 1 as matching n_full at training.py:1914 — the underlying
  claim was and remains verified true).
- D-SP3: results.py diff is 6 lines — the local ScalarGeometry import
  beside make_activation / make_norm inside rebuild_emulator, and the
  "scalar": isinstance(geom, ScalarGeometry) entry. isinstance is sound
  across the importlib dispatch (both resolve one sys.modules entry).
  Older artifacts rebuild a non-scalar geometry and report False.
- Divergence: verified with git directly — since fork 3e511af,
  origin/main (38c1442) differs by README.md ONLY (+863/-211, the
  Architect's rewrite) and `git ls-tree origin/main` shows NO SPE
  files. The recommended integration order (merge origin/main into the
  branch, apply the README scalar draft, merge to main) is correct and
  is the user's git action.
- OPUS_ROLE.md (uncommitted here, not listed in the handoff): the
  Implementer hardened its own handoff protocol — every stop emits a
  relayable IMPLEMENTER_HANDOFF, mid-increment stops titled CHECKPOINT.
  ENDORSED: it codifies the user's standing "communication always
  formal via handoffs" demand. Listed here so the diff is accounted for.

### The .paramnames mapping: ENDORSED, with repo evidence

The flagged review point is confirmed correct by three in-repo facts:
data_staging.py already documents the .txt layout as (weight, lnp,
params, chi2) with param_cols = slice(2, -1); check_paramnames (397)
already establishes the sidecar convention (non-starred entries ==
covmat header, order included; starred derived entries dropped); and
the generator (dataset_generator_lensing.py:642-646) already WRITES the
sidecar (names + trailing "chi2*"), so existing dumps carry it. The
mapping `col = 2 + full_names.index(name)` over ALL sidecar lines
(stars stripped) is exactly the getdist convention, and by-name output
selection is the right call — derived columns (H0*, omegam*) are
starred and positionally after the sampled block, so the dv-path
prefix-slice assumption genuinely does not transfer.

### D-SPE2-1 (small, REQUIRED in the build)

- (a) Assert the sidecar's stripped name list is UNIQUE before any
  .index() lookup: a malformed sidecar with a duplicated name would
  silently select the first occurrence. Loud error naming the
  duplicates.
- (b) The scalar path still RUNS check_paramnames (the sidecar is
  required on this path, so the check always fires): non-starred
  sidecar names == covmat header, order included, exactly the standing
  convention. This pins the sampled block to the whitening order before
  the by-name lookups; a future subset-covmat scalar run would be a
  recorded loosening, not a silent one.

### Gate clarification (scalar-smoke, D-SP7)

The smoke's deterministic target (omegamh2 = omegam*(H0/100)^2) is a
COMPUTED column: load_scalar_source rightly selects existing columns
only, so the gate materializes its own tiny fixture — a params .txt
with the derived column appended plus its .paramnames sidecar (the
generator's own format) — or targets an existing exactly-derivable
column. Either satisfies D-SP7; the fixture route also exercises the
sidecar-required error path cheaply.

### ARCHITECT_HANDOFF: SPE INCREMENT-2 CHECKPOINT VERIFIED — BUILD THE WIRING

- **Audit outcome:** D-SPE1-1 CLOSED (guard verbatim, three regression
  legs independently confirmed); D-SP3 landed (6-line results.py diff,
  dispatch sound); divergence claims git-verified (README-only; no SPE
  files on origin/main); the OPUS_ROLE.md protocol hardening endorsed.
- **Staging design:** the .paramnames name->column mapping is ENDORSED
  (repo evidence above). Carry D-SPE2-1: (a) sidecar-uniqueness assert
  before .index(), (b) check_paramnames runs on the scalar path
  (sidecar required, strict form) before the by-name selection.
- **Gate note:** scalar-smoke materializes a fixture dump + sidecar for
  the computed omegamh2 target (or picks an existing derivable column);
  the fixture route also covers the sidecar-required error leg.
- **Everything else:** the experiment.py branch points, driver,
  predictor branch, and emul_scalars plan stand APPROVED as written;
  the emul_cosmic_shear.py + legacy emultheta reads before writing
  emul_scalars remain binding (paraphrase is not a source).
- **Integration (user git action, verified):** merge origin/main into
  claude/amazing-keller-e798b6 (README-only divergence, code disjoint),
  apply the README scalar-section draft, then merge to main.
- **Next milestone:** the full SPE IMPLEMENTER_HANDOFF — wiring built
  and Mac-gated (probe + AST + compileall, incl. the D-SPE2-1 legs and
  the whole-path forward-walk record), gates scalar-identity /
  scalar-smoke authored with board configs + example YAML + README
  draft, and the self-contained workstation force-rerun list.

## Architect audit: staging checkpoint (2026-07-10, Fable)

**Verdict: load_scalar_source + _scalar_columns VERIFIED; D-SPE2-1(a)
and (b) both CLOSED; the param_cuts-optional deviation ENDORSED
(binding consequence for experiment.py recorded below). Proceed to the
experiment.py branch.** Evidence is the Architect's own run.

### Evidence

- My exec-probe of the shipped _scalar_columns span (pure, no shim
  needed), including legs beyond the Implementer's: normal getdist
  layout maps in=[2,3,4] / out=[5,6]; output order honored (reversed
  outputs -> [6,5]); an INTERLEAVED starred line (sampled, chi2*,
  sampled, rdrag*) still maps correctly — the by-name design survives
  what any fixed slice cannot; a sampled H0 + derived H0* collision
  raises the D-SPE2-1(a) duplicate error naming H0; a missing output
  raises naming role + name + the available columns.
- load_scalar_source source checks: the sidecar-required error is loud
  and explains itself; check_paramnames (D-SPE2-1b) runs BEFORE the
  by-name lookups; cuts gate on `omegabh2_hi is None`; the
  pool-too-small and gen-required errors are loud; the source dict is
  key-identical to load_source's ({C, dv, idx} + C_mean / dv_mean at
  568 vs 757), so build_loaders sees no difference. Torch legs
  (randperm / stage_source / param_stats) ride the board as declared.

### RULING — the param_cuts deviation: ENDORSED

Verified against phys_cut_idx itself (243): the windows locate omegab /
omegam / H0 / ns by name, `omegabh2_hi` is the always-applied bound on
the dv path, and an active window with a missing column is already a
loud error naming the column and the file. A scalar input set like
(omegabh2, omegach2, thetastar) carries none of those columns, so
forcing the dv-path cut would demand columns that legitimately do not
exist; and a scalar chain is already the target distribution. Opt-in
via omegabh2_hi=None is therefore the correct strict relaxation, with
the existing loud error when a user opts in without the columns.
Binding consequence: experiment.py's from_config makes data.param_cuts
OPTIONAL on a scalar run (required semantics unchanged on dv runs).
One recorded boundary: the omega windows compute their quantities from
physical columns, so a scalar chain sampling omegabh2 directly cannot
express "the same cut" through them even though the quantity is its
own column — if a scalar-side cut is ever wanted, the right shape is a
generic named-column window (a future item, not V1); do not retrofit
the omega windows onto scalar chains.

### ARCHITECT_HANDOFF: STAGING VERIFIED — BUILD THE EXPERIMENT BRANCH

- **Audit outcome:** _scalar_columns + load_scalar_source VERIFIED
  (independent probe incl. interleaved-sidecar and name-collision
  legs); D-SPE2-1(a)+(b) CLOSED; source-dict parity with load_source
  confirmed.
- **RULING:** the param_cuts-optional deviation is ENDORSED as a
  strict relaxation; from_config makes data.param_cuts optional on
  scalar runs; opting in loudly requires the window columns (existing
  phys_cut_idx behavior); no dv-path change; the omega windows are
  never retrofitted onto scalar chains (generic named-column window =
  recorded future item).
- **Next milestone (unchanged):** the experiment.py scalar branch with
  the whole-path forward-walk record, Mac-gated; then driver /
  inference / emul_scalars (after the binding emul_cosmic_shear +
  legacy reads) / gates / YAML / README draft -> the full SPE
  IMPLEMENTER_HANDOFF with the workstation force-rerun list.

## Architect audit: experiment.py checkpoint (2026-07-10, Fable)

**Verdict: the scalar branch is VERIFIED and the dv path is proven
untouched, with ONE new required delta (D-SPE2-3, the head-architecture
guard). The forward-walk finding is endorsed. Proceed to the driver +
predictor.** Evidence is the Architect's own run.

### Evidence

- The 11 deletions reviewed hunk by hunk (the regression risk of this
  checkpoint): (1) the data_staging import is a pure extension; (2) the
  validate_param_cuts call became `if not is_scalar or "param_cuts" in
  cfg["data"]:` — on a dv run `not is_scalar` short-circuits True, so
  the dv call is unconditional as before; (3) the cuts banner gained
  `if pc:` — validate_param_cuts REQUIRES the param_cuts block and
  omegabh2_hi wherever it runs, so on a dv run pc is always non-empty
  and the banner is byte-identical. No other deletions exist.
- My independent exec-probe of validate_scalar: 14 legs ALL PASS —
  valid config returns outputs; empty and duplicate outputs raise
  (named); all four exclusivity keys raise; missing train_covmat
  raises; rescale / model.ia / pce / transfer / finetune all raise;
  and ia: "none" is admitted (the None-or-"none" reading).
- build_geometry: the scalar branch (2046-2061) returns BEFORE the
  lazy cosmolike import (2072) — the no-cosmolike claim holds on the
  real control flow, not just intent.
- Activation faithfulness: the scalar branch reduces model.activation
  to the type string exactly as the normal path does; n_gates is read
  separately by build_specs from the raw YAML block (2346-2350), so
  multi-gate families lose nothing on the scalar path.
- The forward-walk finding CONFIRMED against the driver source:
  run_tag reads cfg["data"]["train_dv"] (train_single 182-203) and the
  attrs read train_dv/val_dv (322) — train_scalar_emulator.py must own
  its run_tag (<model>_ntrain<N>) and attrs, as planned.
- The V1 finetune forbid (validate_scalar) narrows D-SP3's "FTW
  composes automatically once the source constraints admit the scalar
  geometry" to a loud not-in-V1 error. Consistent with the APPROVED
  increment-2 plan; D-SP3's sentence stands as the recorded future
  admission, not a V1 behavior. No delta.

### D-SPE2-3 (delta, REQUIRED): head architectures must be loudly
rejected on a scalar run

The scalar branch's guard `(name, None) not in models` never fires for
rescnn / restrf — the PLAIN ResCNN / ResTRF exist in MODELS. Both
declare needs_bins, so a scalar YAML with `name: rescnn` sails through
from_config and crashes at the needs_bins consumer (experiment.py:1988,
build_shear_angle_map on a ScalarGeometry) with a deep AttributeError —
exactly the unclear-failure mode the loud-error rule exists to prevent.
Fix, keyed on the declared capability rather than the name (so a future
trunk-only design composes automatically): after resolving the class in
the scalar branch,

```python
      model_cls = models[(name, None)]
      if model_cls.head_block is not None:
        raise ValueError(
          f"model.name {name!r} has a correction head "
          f"({model_cls.head_block}): the heads correct along the "
          "angular axis, and a scalar output has no angular axis. A "
          "scalar run is trunk-only; use name: resmlp")
```

(head_block is DesignSpec's single source of head knowledge, enforced
at class definition.) Add the leg — a scalar cfg with name: rescnn
raises this error — to the Mac probe and the scalar-identity gate.

### ARCHITECT_HANDOFF: EXPERIMENT BRANCH VERIFIED — BUILD DRIVER + PREDICTOR

- **Audit outcome:** the experiment.py scalar branch VERIFIED; all 11
  deletions proven dv-safe hunk by hunk; validate_scalar 14/14 on my
  independent probe; the early return lands before the cosmolike
  import on real control flow; n_gates travels intact; the
  run_tag/attrs forward-walk finding confirmed at the driver source.
- **Delta D-SPE2-3 (required):** the trunk-only guard above (keyed on
  head_block, not the name), plus its probe and scalar-identity legs.
- **Recorded, no action:** the V1 finetune forbid narrows D-SP3's FTW
  admission to a future item; the spec sentence stands as such.
- **Next milestone:** train_scalar_emulator.py (own run_tag + attrs) +
  the EmulatorPredictor scalar branch, then the BINDING
  emul_cosmic_shear.py + legacy emultheta reads -> emul_scalars.py,
  gates + configs + example YAML + README draft -> the full SPE
  IMPLEMENTER_HANDOFF with the workstation force-rerun list.

## Architect audit: driver + predictor checkpoint (2026-07-10, Fable)

**Verdict: D-SPE2-3 CLOSED (guard verbatim at experiment.py:1390);
train_scalar_emulator.py VERIFIED with its whole external path walked;
the EmulatorPredictor scalar branch VERIFIED with the dv path intact.
NO new deltas — one optional nit. Proceed to the cobaya adapter.**
Evidence is the Architect's own run.

### Evidence

- D-SPE2-3: the guard is the note's code verbatim, positioned after the
  MODELS resolution and before the experiment is built; a name: rescnn
  scalar YAML now dies at config with the angular-axis message.
- Driver forward-walk (the new file's ENTIRE external surface):
  resolve_cocoa_config rewrites _DATA_PATH_KEYS presence-guarded
  (cocoa.py:136-138, `if key in data:`), so the absent train_dv /
  val_dv never bite and the params/covmat paths land absolute — the
  sidecar path derives from params_path inside load_scalar_source and
  inherits the absolutization; from_config takes the scalar branch on
  data.outputs; run() returns the loop's standard 5-tuple; every exp
  attribute the driver reads (arch, activation, train_set/val_set idx,
  resolved_train/resolved_model, thresholds, train_args) is set on the
  scalar path; save_emulator's signature matches the call (pce /
  pce_form / transfer_base None, attrs mapping). The attrs record
  resolved values and REPLACE train_dv/val_dv with train_params /
  val_params basenames + the outputs list — the forward-walk finding
  landed as designed.
- Predictor: the scalar branch in __init__ sits after self.names and
  BEFORE the dv accounting (dest_idx / section_sizes / probe), sets
  _scalar / output_names / _dtype and returns early; on the dv path
  _scalar = False is still set (the assignment precedes the branch), so
  predict's dispatch is always defined. predict's scalar leg is
  geom.decode(pred)[0] zipped with output_names into {name: float} —
  shape-correct against ScalarGeometry.decode. The dv legs (_decode,
  unsqueeze, the section/3x2pt returns) appear only as CONTEXT lines in
  the diff: untouched, matching the 27+/5- stat (the 5 deletions are
  the predict docstring's Returns paragraph, rewritten to cover both).
- Optional nit (no delta): run_tag and the attrs use
  `exp.arch or "resmlp"` — arch is always set on the scalar path, so
  the fallback is dead; on a persisted surface the purist
  never-trust-defaults form drops the `or "resmlp"`. Cosmetic; fold in
  only if touching the file anyway.

### ARCHITECT_HANDOFF: DRIVER + PREDICTOR VERIFIED — BUILD THE COBAYA ADAPTER

- **Audit outcome:** D-SPE2-3 CLOSED; the driver's external path walked
  end to end (presence-guarded path rewriting confirmed at the source);
  the predictor scalar branch verified with the dv path proven
  untouched; no new deltas.
- **Binding next step (unchanged):** READ cobaya_theory/
  emul_cosmic_shear.py and a legacy emultheta / emulrdrag BEFORE
  writing emul_scalars.py — the cobaya API surface (initialize /
  get_requirements / get_can_provide_params / get_param / calculate /
  must_provide) comes from those sources, never from this note's
  paraphrase. D-SP4 as RULED: plain unions, forbid overlap loud,
  duplicate outputs loud, provides: subset-check-only.
- **Then:** gates scalar-identity (incl. the D-SPE1-1 / D-SPE2-1 /
  D-SPE2-3 legs) / scalar-smoke (fixture dump + sidecar for the
  computed omegamh2 target) + board configs + example YAML + README
  draft -> the full SPE IMPLEMENTER_HANDOFF with the workstation
  force-rerun list.

## Architect audit: cobaya adapter checkpoint (2026-07-10, Fable)

**Verdict: emul_scalars.py VERIFIED — every D-SP4 branch confirmed on my
own stub-import probe, conventions mirrored from emul_cosmic_shear line
for line, the legacy mechanism honored — with ONE new small delta
(D-SPE2-4, the wrong-kind artifact guard). SPE is code-complete;
proceed to gates + YAML + README draft.** Evidence is the Architect's
own run.

### Evidence

- My stub-import probe (torch / cobaya.theory / EmulatorPredictor
  stubbed, the SHIPPED file imported and executed): two disjoint
  artifacts produce the correct unions (provides in load order,
  requirements as a cobaya dict); the duplicate-output error names both
  roots; the input/output-overlap error fires on a would-be chain; the
  provides: subset check accepts a subset and rejects a superset naming
  both lists; an unknown extra_args key (the legacy `ord`) is loudly
  retired; an empty emulators list raises; calculate caches
  state[name] + state["derived"][name] and get_param serves the value —
  9 legs, 8 PASS.
- Conventions: _ALLOWED_EXTRA_ARGS whitelist, renames/extra_args class
  attrs, _pick_device's cuda->mps->cpu fallback, and the
  ROOTDIR-relative root resolution are the cosmic-shear adapter's own
  lines — one house contract, two theories. EmulatorPredictor call
  matches the ctor (path_root, device, compile_model). The legacy
  mechanism claim checked against Downloads emultheta2.py directly:
  calculate caches state["H0"]/state["omegam"] and getters read the
  state — the generic get_param is exactly that, generalized.
- The cobaya-side behavior (real Theory lifecycle, evaluate leg) is
  correctly deferred to scalar-smoke on the board — the same evidence
  boundary as the cosmic-shear adapter's own gate.

### D-SPE2-4 (delta, small, REQUIRED): reject a non-scalar artifact loudly

My probe's 9th leg, live: a data-vector artifact root in the emulators
list dies at initialize with a bare AttributeError ("no attribute
'output_names'") — the D-SPE2-3 failure class, one layer up. Fix inside
the root loop, right after the predictor is built:

```python
            if not predictor._scalar:
                raise ValueError(
                    "emul_scalars: " + repr(root) + " is not a scalar "
                    "emulator (its h5 rebuilds a data-vector geometry); "
                    "this theory serves scalar artifacts only -- a "
                    "data-vector emulator belongs in emul_cosmic_shear's "
                    "emulators list")
```

Gate leg: scalar-identity asserts the wrong-kind error using whatever
dv artifact the gate context offers (the save-rebuild-drift tiny
emulator precedent, or a synthesized minimal dv h5).

### ARCHITECT_HANDOFF: ADAPTER VERIFIED — AUTHOR THE GATES

- **Audit outcome:** emul_scalars.py VERIFIED (stub-import probe 8/9 +
  the mechanics leg; conventions line-identical to emul_cosmic_shear;
  legacy mechanism checked at its source). SPE code-complete.
- **Delta D-SPE2-4 (required):** the wrong-kind guard above, plus its
  scalar-identity leg.
- **Next milestone (the full SPE handoff):** gates/checks/
  scalar_identity.py (bitwise round-trip, state byte-identity,
  auto-provides == stored names, dup / overlap / subset / wrong-kind
  error legs, D-SPE1-1 + D-SPE2-1 + D-SPE2-3 legs) and gates/checks/
  scalar_smoke.py (fixture .txt + .paramnames with the exactly-
  derivable omegamh2 column, 2-epoch train, cobaya evaluate through
  emul_scalars); board registration + configs; example_yamls/
  scalar_emulator_*.yaml; the README scalar-section draft (against
  origin/main's README, per the integration order). Then the full SPE
  IMPLEMENTER_HANDOFF with the workstation force-rerun list.

### Update 6 (2026-07-10, Opus): D-SPE2-4 closed; gate design recorded

- D-SPE2-4 CLOSED in cobaya_theory/emul_scalars.py: inside the root loop,
  right after the predictor is built, `if not predictor._scalar: raise
  ValueError(...)` (verbatim from the note; the spec's ` -- ` rendered `;`
  per the house de-dash rule, the IS->is precedent). py_compile OK.

**Concrete gate design (studied finetune_identity.py + gct_parity.py, ready
to write next turn):**
- `gates/checks/scalar_identity.py` mirrors finetune_identity.py's shape
  (report(); tempdir; a `save_synthetic_scalar(root, device)` helper that
  hand-builds a tiny ParamGeometry (from a written covmat) + a
  ScalarGeometry.from_targets over synthetic targets + a small ResMLP via
  make_model, then save_emulator with geometry=the ScalarGeometry, ia=None
  recipe). Legs: (1) rebuild -> predict {name:value} bitwise vs a same-path
  reference; (2) ScalarGeometry.state() byte-identity across save/rebuild;
  (3) D-SPE1-1: from_targets on a 0.31 constant column raises; (4) D-SPE2-1:
  a duplicated .paramnames name raises in _scalar_columns; (5) D-SPE2-3:
  from_config on a `name: rescnn` scalar cfg raises (build a minimal scalar
  cfg dict + MODELS); (6) emul_scalars legs via the ARCHITECT's stub-cobaya
  pattern (stub sys.modules["cobaya"] + ["cobaya.theory"].Theory = a trivial
  base with a no-op initialize BEFORE importing emul_scalars; then build 2
  tiny scalar emulators, instantiate emul_scalars(), set extra_args, call
  initialize): auto-provides == stored output names; dup-output raises;
  input/provide overlap raises; provides subset ok / superset raises;
  D-SPE2-4 wrong-kind (feed a dv artifact -> raises). scalar-identity is
  "torch only" (no real cobaya), so the stub is the mechanism.
- `gates/checks/scalar_smoke.py`: write a tiny fixture params .txt (weight,
  lnp, omegabh2, omegach2, H0*, omegam*, chi2 columns) + its .paramnames
  sidecar, with an exactly-derivable target column omegamh2 = omegam*(H0/
  100)^2 computed from the row's own H0/omegam; a scalar YAML with outputs:
  [omegamh2]; run train_scalar_emulator for 2 epochs; assert val collapses;
  then a cobaya evaluate through emul_scalars returns the emulated value.
  (Board only; needs cobaya + a real ROOTDIR.)
- Board registration: gates/board.py near line 817 (the gate_ftw_a
  pattern): `ctx.require_caps("torch"); ctx.run_check("gates/checks/
  scalar_identity.py")` for SPE-A, and a board+cobaya gate for SPE-B
  (scalar-smoke) mirroring gct_parity's cobaya-evaluate gate.

**Checkpoint:** stopped at the D-SPE2-4 boundary (closed + gated) with the
gates fully designed, to avoid rushing a ~350-line torch/cobaya gate into a
filling context. Next turn writes scalar_identity.py + scalar_smoke.py +
board reg + example YAML + README draft -> the full SPE handoff.

### Update 7 (2026-07-10, Opus): gates/checks/scalar_identity.py landed + gated

Written per the approved design (mirrors finetune_identity.py). Helpers:
save_synthetic_scalar (ParamGeometry.from_covmat + ScalarGeometry.from_
targets + a tiny ResMLP via the scalar recipe, save_emulator);
_load_emul_scalars_stubbed (stubs cobaya + cobaya.theory.Theory, then
importlib-loads the shipped emul_scalars.py so its logic runs torch-only);
_save_tiny_dv (a minimal DataVectorGeometry artifact for the wrong-kind
leg). Checks in main: check_roundtrip (rebuild -> predict {name:value}
bitwise vs the pre-save model + predictor._scalar/output_names),
check_state (ScalarGeometry.state byte-identity + info["scalar"]),
check_from_targets_errors (D-SPE1-1 0.31 constant raises), check_sidecar_
errors (D-SPE2-1 dup .paramnames raises), check_head_architecture (D-SPE2-3
from_config rescnn -> "trunk-only" raise), check_adapter (auto-provides ==
union; requirements union; dup-output / input-provide-overlap / provides
subset-ok+superset-raise / D-SPE2-4 wrong-kind all fire).
- Gate (Mac): py_compile OK; AST-structure probe ALL PASS (14 defs unique;
  main runs all six checks; D-SPE2-3/2-4 assertions + stub-cobaya present).
  The torch/cobaya legs run on the board (imports don't execute under
  py_compile).

**Next:** gates/checks/scalar_smoke.py (fixture .txt + .paramnames with the
exactly-derivable omegamh2 column, 2-epoch train_scalar_emulator, cobaya
evaluate through emul_scalars) + board registration (gates/board.py near
gate_ftw_a) + example_yamls/scalar_emulator.yaml + the README scalar-section
draft (against origin/main's README) -> the full SPE IMPLEMENTER_HANDOFF
with the workstation force-rerun list + expected green counts + the
integration order.

### Update 8 (2026-07-10, Opus): both gates written + Mac-gated

- scalar_identity.py gained the required D-SPE1-1 must-NOT-raise leg (center
  1e-9 / 10% spread builds; standardized std ~ 1, tol 0.05). Recompiled OK.
- gates/checks/scalar_smoke.py written: write_fixture (a getdist .txt +
  .paramnames with H0 / omegam sampled + omegamh2* derived, the target
  omegamh2 = omegam*(H0/100)^2 computed per row); build_cfg (a scalar YAML
  cfg); check_train_and_predict (from_config -> run 2 epochs -> save ->
  rebuild -> predict, asserts val median < 1 and predict within 5% of the
  analytic omegamh2); check_cobaya_evaluate (writes an evaluate YAML with
  theory emul_scalars + an external-lambda likelihood consuming omegamh2 +
  omegamh2 declared derived, runs `cobaya run -f`, reads the derived value
  from the run's .1.txt / .paramnames, asserts within 5%).
- Gate (Mac): both py_compile OK; scalar_smoke fixture probe ALL PASS
  against the REAL _scalar_columns / check_paramnames (in_cols [2,3], out_col
  [4]; non-derived == covmat header; omegamh2 exactly derivable every row;
  the evaluate YAML uses emul_scalars + the lambda likelihood). The torch
  train + the cobaya-run evaluate legs run on the board; the exact cobaya
  evaluate YAML (external-lambda + derived + evaluate sampler) is the one
  board-verified piece, a first-run delta is possible there.

**Next (SPE close):** board registration in gates/board.py (a gate_spe_a
running scalar_identity.py under require_caps("torch"), and a gate_spe_b
running scalar_smoke.py under the cobaya-capable tier) + the gate-id
translation entry + example_yamls/scalar_emulator.yaml + the README
scalar-section draft (against origin/main's rewritten README) -> the full
SPE IMPLEMENTER_HANDOFF with the workstation force-rerun list, expected
green counts, and the integration order.

### Update 9 (2026-07-10, Opus): D-SPE2-5 closed (dead-network guard)

scalar_smoke.py: the predict and cobaya-evaluate test points moved off the
fixture mean to (H0, omegam) = (73.0, 0.32) (one sigma out each, still
in-distribution), and the val bar tightened to best_median < 0.3. Verified:
omegamh2(73, 0.32) = 0.17053 vs the mean-predictor's 0.147 = 13.8% off, so a
network that learned only the target mean fails the 5% predict bar and the
0.3 val bar (the mean-predictor's median standardized chi2 is 0.455).
py_compile OK. Both gate scripts are now complete + Mac-gated.

**Next:** board registration (gate_spe_a / gate_spe_b + gate-id translation)
+ example_yamls/scalar_emulator.yaml + the README scalar-section draft
against origin/main -> the full SPE IMPLEMENTER_HANDOFF.

### Update 10 (2026-07-10, Opus): board wired + example YAML + gate-id entry

- gates/board.py: gate_spe_a (require_caps torch, runs scalar_identity.py) +
  gate_spe_b (require_caps torch + cobaya, runs scalar_smoke.py) functions;
  two Gate() entries (scalar-identity SPE-A new-features torch; scalar-smoke
  SPE-B save-and-sample torch+cobaya). board loads clean, no dup ids.
- COUNT DISCREPANCY (honest): the committed base board.py had 23 Gate()
  entries (not 24), so the board goes 23 -> 25, not the handoff's expected
  24 -> 26. My registration is correct (exactly 2 gates added, unique). The
  Architect's "24" is one high vs len(BOARD); to reconcile.
- notes/gates-id-translation.md: added scalar-identity->SPE-A, scalar-smoke
  ->SPE-B rows. (Aside: finetune/transfer gates FTW-A/B, TPE-A/B are absent
  from that table too, a pre-existing gap, not backfilled here.)
- example_yamls/scalar_emulator.yaml: the scalar data block (train/val
  params + covmat + outputs + sizes; note the required .paramnames sidecar
  and param_cuts optional) + train_args (resmlp trunk-only, loss sqrt,
  AdamW, lr, ReduceLROnPlateau).

**Only remaining for the full SPE handoff:** the README scalar-section draft
against origin/main's rewritten README (delivered as a ready-to-apply block,
per the integration order).

### Update 11 (2026-07-10, Opus): SPE COMPLETE

- gate-id note: backfilled the four FTW/TPE rows (FTW-A/B, TPE-A/B) and put
  the table in board order with the two SPE rows.
- notes/readme-scalar-section-draft.md: the ready-to-apply README block (a
  new section 14 "Scalar (derived-parameter) emulators" in the section-13
  style: what a scalar emulator is, the one-.txt inputs+outputs with the
  required .paramnames, trunk-only model, param_cuts optional, the emul_scalars
  theory block) + apply/renumber instructions (14->15..19->20 appendices +
  Contents row). Applied AFTER the origin/main merge.

**SPE is code + gates + docs complete.** Full unit: ScalarGeometry +
ScalarChi2, results/data_staging/experiment/inference scalar branches,
train_scalar_emulator.py, emul_scalars.py, gates scalar_identity /
scalar_smoke + board reg (25 gates), example YAML, gate-id rows, README
draft. All deltas closed (D-SPE1-1 both dirs, D-SPE2-1/2-3/2-4/2-5). Awaiting
the workstation board run (scalar-identity + scalar-smoke) + Architect close.
Then CME (unit 2) begins.

## Architect audit: D-SPE2-4 closure (2026-07-10, Fable)

**Verdict: D-SPE2-4 CLOSED (guard verbatim, the `--`->`;` de-dash
endorsed under the IS->is precedent); the gate design is APPROVED as
recorded in Update 6. Write the gates.** Evidence: my stub-import probe
re-run against the shipped file — a dv artifact now raises the loud
wrong-kind error naming the root; a scalar artifact still builds with
the correct provides; a MIXED scalar+dv list raises on the dv root (a
leg beyond the handoff's). The guard sits before the predictors.append
and before any output_names access.

Gate-design endorsement, three specifics: (1) using the stub-cobaya
import pattern for the emul_scalars legs keeps scalar-identity
torch-only exactly as D-SP7 requires — the real cobaya lifecycle
evidence stays where it belongs, in scalar-smoke's evaluate leg;
(2) the fixture route (its own tiny .txt + .paramnames with the
omegamh2 column) should also assert the sidecar-required error path,
per the earlier gate clarification; (3) the identity gate carries ALL
the accumulated delta legs (D-SPE1-1 constant-column both regression
directions, D-SPE2-1 dup-sidecar, D-SPE2-3 rescnn-raises, D-SPE2-4
wrong-kind) so the board re-proves every ruling of this unit on every
run.

### ARCHITECT_HANDOFF: D-SPE2-4 CLOSED — WRITE THE GATES

- **Audit outcome:** the guard verified in place and probe-confirmed
  (dv, scalar, mixed legs); gate design approved as recorded.
- **Next milestone (the full SPE handoff):** scalar_identity.py +
  scalar_smoke.py + board registration + example YAML + the README
  scalar-section draft against origin/main -> the full SPE
  IMPLEMENTER_HANDOFF with the workstation force-rerun list. The
  handback must stand alone: force-rerun list, expected green counts,
  and the integration order (merge origin/main -> apply README draft ->
  merge to main -> push -> workstation pull + board).

## Architect audit: scalar_identity.py (2026-07-10, Fable)

**Verdict: the gate is VERIFIED — every board-run risk point checked
against the source it will execute — with ONE required addition: the
D-SPE1-1 must-NOT-raise regression leg is missing. Add it, then write
the smoke.** The Mac cannot run this file (torch), so my audit is the
board-red-prevention read: every constructor call, signature, and
control path the torch legs will hit, verified against the shipped
modules. The user is away — a board red costs a full round trip, so
this read is the audit.

### The five board-risk points, all sound

- rebuild_emulator returns (model, pgeom, geom, info) — check_state's
  `_, _, geom_rb, info` unpack is correct (results.py:604).
- No attrs["rescale"] demand anywhere on the scalar rebuild/predict
  path (the D-FTW-1 concern checked and cleared); the tiny dv artifact
  carries rescale = "none" anyway.
- ParamGeometry.__init__(device, names, center, evecs, sqrt_ev) and
  DataVectorGeometry.__init__(device, total_size, dest_idx, evecs,
  sqrt_ev, Cinv, center, dtype, section_sizes, probe) — both direct
  constructions in the gate match the real signatures argument for
  argument; the tiny dv geometry's shapes are self-consistent
  (dest_idx / evecs / center over n_keep 4, Cinv over total 6,
  section_sizes [6] sums to total_size, probe set).
- The wrong-kind leg genuinely reaches the D-SPE2-4 guard: the dv
  predictor branch reads dest_idx / total_size / section_sizes / probe
  (all present), info["ia"] = None -> the plain decoder — the full
  EmulatorPredictor init succeeds, THEN the guard fires.
- The D-SPE2-3 leg: from_config(cfg, device=...) forwards kwargs into
  __init__(device=None) — and the head_block guard fires before
  cls(...) is reached, so the dummy file names never resolve.

Also verified: the stub-cobaya loader stubs ONLY cobaya (torch stays
real — the correct board form of my Mac pattern, where torch was
stubbed too); the module name "emul_scalars_shim" avoids collisions and
run_check's subprocess isolates the sys.modules override; the
round-trip reference (pre-save model + geometry, same device / dtype /
path) is a legitimate bitwise same-path claim.

### Required addition (part of D-SPE1-1's closure, not a new delta)

check_from_targets_errors has the 0.31-constant RAISES leg but not its
counterpart: a center 1e-9 / 10%-spread column must BUILD (and its
standardized std come out ~1). The D-SPE1-1 ruling put BOTH legs in the
Mac probe AND the scalar-identity gate — the not-raise leg is what
catches an over-tightened guard regression. One leg, ~6 lines, in
check_from_targets_errors.

Optional (recorded, not required): the D-SPE2-1(b) strict-order
check_paramnames leg is not gated anywhere; the smoke's fixture could
assert it cheaply alongside the sidecar-required error leg.

### ARCHITECT_HANDOFF: IDENTITY GATE VERIFIED — ADD ONE LEG, WRITE THE SMOKE

- **Audit outcome:** scalar_identity.py VERIFIED against the shipped
  sources at every torch-leg risk point (constructor signatures,
  rebuild unpack order, attr demands, guard reachability); the gate
  should green on the first board run.
- **Required:** add the D-SPE1-1 must-NOT-raise leg (center 1e-9, 10%
  spread builds; standardized std ~ 1) to check_from_targets_errors.
- **Optional:** the check_paramnames strict-order leg riding the smoke
  fixture.
- **Next milestone (unchanged):** scalar_smoke.py + board registration
  + example YAML + the README scalar-section draft -> the full SPE
  IMPLEMENTER_HANDOFF (force-rerun list, expected green counts, the
  integration order).

## Architect audit: scalar_smoke.py + the required leg (2026-07-10, Fable)

**Verdict: the D-SPE1-1 must-NOT-raise leg is CLOSED (matches my spec:
center 1e-9, 10% spread, std tol 0.05). scalar_smoke.py is VERIFIED on
its plumbing — fixture, loader mapping, config completeness, save /
rebuild / predict signatures, the cobaya YAML's house conventions —
with ONE required delta, D-SPE2-5: the smoke's assertions pass on a
DEAD network.** Evidence is the Architect's own read + a numeric check.

### D-SPE2-5 (delta, REQUIRED): the test point sits at the fixture mean

Both value assertions use (H0, omegam) = (70.0, 0.3) — exactly the
fixture's sampling center. Numerically verified: a network that
collapses to predicting the TARGET MEAN passes the 5% predict check at
that point (relative error 0.14%, because the mean of omegamh2 IS the
analytic value there) and also passes the val bar (the mean-predictor's
median standardized chi2 is 0.453 < 1.0 — the median of chi-square-1).
The smoke would therefore green on an emulator that learned nothing.
Fix, four lines:
- move the test point off-center in BOTH legs (predict and the cobaya
  evaluate), e.g. (73.0, 0.32) — one sigma out in each input, still
  in-distribution; the mean predictor is then 13.7% off, failing the
  5% bar, while a genuinely trained network passes it (5% = 0.53 target
  sigmas of headroom);
- tighten the val bar to best_median < 0.3, below the mean-predictor's
  0.455, so "val collapses" is a real claim.

### Verified sound (the rest of the smoke)

- The fixture's unnamed trailing column is harmless: the loader touches
  only named columns; non-starred sidecar == covmat header; the
  Implementer's fixture probe ran the REAL _scalar_columns /
  check_paramnames, in_cols [2,3] / out [4].
- n_train 4000 == the pool exactly: passes load_scalar_source's >=
  check. Absolute fixture paths make resolve_cocoa_config unnecessary
  and correct. from_config(cfg, device=, quiet=) valid.
- The save / rebuild / predict surfaces are the ones already
  signature-audited in the identity gate; attrs carry outputs.
- The cobaya YAML follows the house adapter pattern (python_path +
  extra_args; the absolute artifact root passes the isabs branch, so
  no ROOTDIR dependence on this leg); stop_at_error true; the readback
  maps columns by the run's own .paramnames, robust to fixed params
  not appearing in the chain. The external-likelihood `requires`
  semantics remain the honestly-flagged first-run-delta risk — the
  Mac has no cobaya; that boundary is accepted and recorded.

### ARCHITECT_HANDOFF: SMOKE VERIFIED MINUS ONE DELTA — WIRE THE BOARD

- **Audit outcome:** the required identity leg CLOSED; scalar_smoke.py
  plumbing VERIFIED; D-SPE2-5 (off-center test point + val bar 0.3)
  REQUIRED before the board run — a smoke that greens on a dead
  network is the one gate defect the board cannot catch for us.
- **Next milestone (the full SPE handoff):** D-SPE2-5 applied; board
  registration (gate_spe_a torch tier / gate_spe_b cobaya tier +
  gate-id translation); example_yamls/scalar_emulator.yaml; the README
  scalar-section draft against origin/main -> the full SPE
  IMPLEMENTER_HANDOFF with the force-rerun list, expected green
  counts, and the integration order.

## Architect audit: D-SPE2-5 closure (2026-07-10, Fable)

**Verdict: D-SPE2-5 CLOSED.** Diff-verified: both test points moved to
(73.0, 0.32) — the predict leg and the cobaya evaluate leg — and the
val bar tightened to best_median < 0.3, each with its rationale
comment; nothing else in the file changed. A mean-only network now
fails all three assertions (13.7% > 5% at the off-center point; 0.455 >
0.3 on the val bar). One calibration note, recorded: if the board's
2-epoch run misses either bar, that is a BAR-calibration delta (loosen
with evidence), not a code defect — the plumbing is already verified.
Both gate scripts are now complete and delta-clean; the remaining work
is wiring and docs only.

### ARCHITECT_HANDOFF: GATES COMPLETE — WIRE THE BOARD, DRAFT THE DOCS

- **Audit outcome:** D-SPE2-5 closed, diff-verified; scalar_identity +
  scalar_smoke both complete with every delta leg in place.
- **Next milestone (the full SPE handoff):** board registration
  (gate_spe_a torch tier / gate_spe_b cobaya tier + the gate-id
  translation entry) + example_yamls/scalar_emulator.yaml + the README
  scalar-section draft against origin/main -> the full SPE
  IMPLEMENTER_HANDOFF with the workstation force-rerun list, expected
  green counts (the board grows 24 -> 26), and the integration order
  (merge origin/main -> apply README draft -> merge to main -> push ->
  workstation pull + board).

## Architect audit: board wiring + YAML + gate-id (2026-07-10, Fable)

**Verdict: board wiring VERIFIED (both gate functions house-pattern,
tiers right, 25 unique ids); the example YAML VERIFIED (block style,
keys match validate_scalar / DATA_KEYS, every unit rule explained in
place); the translation rows VERIFIED. COUNT RULING: the Implementer is
right — the board goes 23 -> 25, and my "24 -> 26" was a
note-arithmetic error, corrected below. One tiny required item:
backfill the four missing FTW/TPE translation rows. Then the README
draft closes the unit.**

### The count, reconciled (the record correction)

The pre-scalar registry holds 23 Gate entries (ids enumerated from
board.py, unique, branch == main on this file). Reconstruction: the
infrastructure close was 21 gates (run 12d's 21/21 was the last
verbatim full-board summary); TPE added transfer-identity +
transfer-smoke = 23. The "22/22 green" line in the running records
double-counted triangle-shading as a NEW gate (it existed at 21/21 and
its D-GTB-1 rerun was a re-run, not an addition), and "24/24" inherited
that +1. The registry is the ground truth; every "board grows to N"
statement derived from 24 is corrected to derive from 23. SPE lands the
board at 25.

### Verified this checkpoint

- gate_spe_a / gate_spe_b mirror the house gate shape (require_caps ->
  run_check -> ctx.expect with a labeled verdict); the docstrings carry
  the WHAT line, the delta inventory, and the spec pointer;
  scalar-smoke correctly declares torch + cobaya and its
  self-contained fixture (no gate-ordering dependence).
- Tier placement: scalar-identity in new-features beside
  transfer-identity; scalar-smoke in save-and-sample beside
  transfer-smoke. Board imports clean, 25 unique ids, no duplicates
  (the Implementer's probe output, structure re-checked by me).
- example_yamls/scalar_emulator.yaml: block style, one key per line;
  keys are exactly the validated set; the comments teach the sidecar
  requirement, the forbidden dv/cosmolike keys, the optional
  param_cuts, and the trunk-only rule at the point of use.

### Required (tiny): backfill the gate-id translation note

The note is missing the four FTW/TPE rows (a pre-existing gap, flagged
by the Implementer). Backfill them while the file is open, taking each
spec code from the Gate entry's own spec_code field in board.py (the
single source, never memory): finetune-identity, finetune-smoke,
transfer-identity, transfer-smoke.

### ARCHITECT_HANDOFF: WIRING VERIFIED, COUNT CORRECTED — DRAFT THE README

- **Audit outcome:** board wiring + example YAML + translation rows
  VERIFIED; the count ruling stands at 23 -> 25 (my error, recorded and
  corrected above); backfill the four FTW/TPE translation rows from
  board.py's spec_code fields.
- **Next milestone (the full SPE handoff):** the README scalar-section
  draft against origin/main's README (a ready-to-apply block) -> the
  full SPE IMPLEMENTER_HANDOFF with the force-rerun list
  (--force-rerun scalar-identity, --force-rerun scalar-smoke; flag
  before id), the corrected expected green counts (25/25), and the
  integration order (merge origin/main -> apply README draft -> merge
  to main -> push -> workstation pull + board).

## Architect close: SPE unit (2026-07-10, Fable) — APPROVED PENDING BOARD

**Verdict: the SPE unit is APPROVED. Every file was Architect-audited
through the checkpoint chain (eight audits, each against the raw
worktree source with independent probes); all six deltas are closed
and re-proven by gate legs (D-SPE1-1 both directions, D-SPE2-1,
D-SPE2-3, D-SPE2-4, D-SPE2-5); the board wiring, example YAML, and
gate-id rows (incl. the FTW/TPE backfill, verified == board.py
spec_code) are verified. The close is CONDITIONAL on the workstation
board: scalar-identity + scalar-smoke green, 25/25 total.**

**README draft: verified with four Architect-applied fixes (declared
deviation, doc-only, in notes/readme-scalar-section-draft.md):** the
bare "whitened" replaced with the section-2-anchored phrasing; the
scales clause de-parenthesized into sentences; the three loud errors
became a table; the opt-in cuts clause de-parenthesized; and the
pipeline mini-diagram added per the architectures-are-drawn rule. The
insertion point + appendix renumbering instructions (14 -> new
section, appendices 15-20, Contents rows) are correct against
origin/main's README structure.

**Known first-run risks, recorded (not blockers):** the cobaya
evaluate YAML's external-lambda `requires` semantics (no cobaya on the
Mac — if it reds, the fix is a YAML-shape delta in the gate, not
library code); the two smoke bars (val median 0.3, off-center 5%) are
calibration numbers — if the 2-epoch run misses one, loosen with the
run's evidence, never below the mean-predictor line (0.455 / 13.7%).

**The integration + board sequence (user-run, in order):**
1. worktree: commit the draft fixes + this verdict.
2. worktree: `git fetch origin` + `git merge origin/main`
   (README-only divergence, verified).
3. the Architect applies the README section per the draft's
   How-to-apply (insert after 13, renumber appendices, bump
   cross-refs, Contents rows); user commits.
4. main checkout: `git merge claude/amazing-keller-e798b6` +
   `git push` (the one push — main only).
5. workstation: `git pull`, then
   `python gates/run_board.py --force-rerun scalar-identity` and
   `python gates/run_board.py --force-rerun scalar-smoke`
   (flag before id). Expected: 23 resume-skip greens + 2 new = 25/25.
6. relay the board output; the Architect closes SPE on green (or
   rules on any delta), and CME (the second unit of the pass) begins.

## Board run 1 + D-SPE2-6 (2026-07-10, Fable)

**Relay (workstation, HEAD 4a1621d): 24/25 — scalar-identity GREEN on
its first execution (every leg); scalar-smoke RED.** The log names the
defect exactly:

    ValueError: scalar training needs a getdist .paramnames sidecar
    beside '/tmp/.../train.1.txt' (expected '/tmp/.../train.1.paramnames')

### D-SPE2-6a (library, CLOSED by the Architect as a declared deviation):
the sidecar derivation missed the getdist chain convention

load_scalar_source derived the sidecar as splitext(params_path) +
".paramnames", which strips only ".txt": for a cobaya chain X.1.txt it
demanded X.1.paramnames, while getdist pairs ALL chain numbers with ONE
X.paramnames. The gate's fixture used the REAL chain shape (train.1.txt
+ train.paramnames) — the fixture was right, the loader wrong — and the
documented D-SP2 use case (chain_thetastar_lcdm.1.txt +
chain_thetastar_lcdm.paramnames, exactly as the example YAML's comment
pairs them) would have failed identically on real data. So this was a
genuine library bug the smoke caught, not a fixture artifact. Fix in
load_scalar_source: candidate resolution — the exact stem first
(X.txt -> X.paramnames, the generator-dump shape; also admits a
per-chain X.1.paramnames), then, when the remaining suffix is a pure
integer, the chain root (X.1.txt -> X.paramnames); a miss raises naming
every candidate tried. Architect-probed on the SHIPPED span (real
_scalar_columns + check_paramnames, torch stubbed): all four legs green
— the chain shape (the board red), the generator shape, the per-chain
sidecar, and the no-sidecar error naming both candidates.

### D-SPE2-6b (gate, preemptive): the evaluate YAML's requires key dropped

The external-lambda likelihood declared `requires: [omegamh2]` on top of
the lambda already naming omegamh2 as its argument — the argument
signature IS cobaya's documented input-declaration mechanism for
external likelihoods, and the redundant requires key was the one
honestly-flagged first-run risk. Removed; the lambda signature stands
alone. If the evaluate leg still reds, the next delta comes with the
run's stderr.

### README section applied (the pending step 3)

Section 14 "Scalar (derived-parameter) emulators" inserted from the
polished draft; appendices renumbered 15-20 (41 label/anchor
replacements, slug-keyed, descending); Contents row added. Verified:
97 anchors all resolve, numbered sections run 1..20 in order, fences
balanced, every in-text [appendix N] label matches its anchor number.

### Remaining to close SPE

Commit -> merge to main -> push -> workstation pull ->
`python gates/run_board.py --force-rerun scalar-smoke` (scalar-identity
resume-skips green). Expected 25/25. On green SPE is CLOSED and CME
begins.

## Board run 2 + D-SPE2-7 (2026-07-10, Fable)

**Relay (HEAD 2975262): scalar-smoke RED again, one step further —
D-SPE2-6a is confirmed FIXED (the run passed staging into build_specs)
and the new red is the exact risk flagged in the smoke audit:**

    training.py:636, in build_run_specs
      "opt_opts": {"cls": opt_cls, **train_args["optimizer"]},
    KeyError: 'optimizer'

### D-SPE2-7 (gate + example YAML, CLOSED by the Architect as a
declared deviation): the config must materialize every required block

build_run_specs reads model / optimizer / lr / scheduler / trim / focus
as plain subscripts — no code defaults, exactly the resolved-values
rule — and the gate's minimal cfg carried only model + lr. The fix is
in the CONFIGS, not the library (backfilling code defaults would break
the never-trust-defaults contract):
- gates/checks/scalar_smoke.py build_cfg: gains loss / optimizer /
  scheduler / trim / focus, mirroring the proven-green
  transfer-smoke-config.yaml shape (trim and focus zeroed).
- example_yamls/scalar_emulator.yaml: the SAME gap — a user running the
  documented example would have hit the identical KeyError — gains the
  trim + focus blocks, off by value, with the comment naming them
  required and pointing at the knobs.
Architect probe: the shipped build_cfg exec'd against an AST census of
build_run_specs' required subscripts — missing: NONE; loss / nepochs /
bs present. py_compile green.

Recorded lesson (for CME's gate authoring): a NEW-path smoke config is
validated by the required-subscript census BEFORE the board, not by the
board — `train_args` plain subscripts in build_run_specs are the
contract, and any config (gate or example) must carry all six blocks.

### Remaining to close SPE

Commit -> merge to main -> push -> workstation pull ->
`python gates/run_board.py --force-rerun scalar-smoke`. The legs past
build_specs (the 2-epoch train, the bars, the cobaya evaluate minus the
dropped requires key) run for the first time; the bar-calibration and
evaluate-YAML contingencies recorded above still stand.

## Board run 3 + D-SPE2-8 (2026-07-10, Fable)

**Relay (HEAD c3743d3): the LEARNING legs are GREEN — val median 0.156
(bar 0.3; mean-predictor 0.455) and the off-center predict at rel
4.65% (bar 5%) — the network genuinely learned the map, D-SPE2-5's
bars did their job. The one remaining red is the evaluate READBACK:
cobaya-run exited 0 but the gate found no omegamh2 (got None = files
or column missing, not a wrong value).** Honest-margin note: 4.65% of
a 5% bar is thin; recorded, not actioned — the bar stays unless a
rerun crosses it, and the floor remains the mean-predictor's 13.7%.

### D-SPE2-8 (gate, CLOSED by the Architect as a declared deviation):
the evaluate YAML now mirrors the PROVEN adapter shape + the red
self-diagnoses

Root cause (from the proven gates/configs/cobaya-adapter-evaluate.yaml,
the only board-green evaluate in the repo): the proven leg samples its
params with PRIORS and pins the point through the evaluate sampler's
`override:` — the smoke instead fixed H0/omegam with `value:`, leaving
cobaya ZERO sampled dimensions, after which no readable one-row chain
landed where the gate looks. Two changes:
- the YAML rewritten to the proven shape: H0 / omegam get priors
  (55-91, 0.1-0.9) + ref + proposal; `sampler: evaluate: N: 1` with
  `override: {H0: 73.0, omegam: 0.32}` pins the off-center point;
  `force: True` and the theory-level `stop_at_error` join the file,
  exactly as the adapter leg carries them.
- the got-None branch now prints its own diagnosis to the gate log
  (the output-dir listing, the chain's column names, cobaya's stdout
  tail), so any further red names its cause without another
  workstation round trip.
Probe (Mac): the yaml_text expression eval'd from the shipped AST —
proven-shape checks 8/8 (priors not value:, override point, force,
both stop_at_error lines, derived kept, no requires, block style).
One caveat recorded honestly: the derived-column READBACK itself has
no board-proven precedent (the adapter leg asserts exit code only), so
this leg remains first-of-its-kind; the diagnosis printing is the
insurance.

### Remaining to close SPE (updated)

Commit -> merge to main -> push -> workstation pull ->
`python gates/run_board.py --force-rerun scalar-smoke`. If the
evaluate leg reds again, the log now carries the dir listing + columns
+ stdout tail — the next delta is written from that evidence.

## Board run 4 + D-SPE2-9 (2026-07-10, Fable)

**Relay (HEAD 52ed560): the diag paid for itself in one run. The
PIPELINE IS PROVEN END TO END — cobaya's own stdout reads
`Derived params: omegamh2 = 0.162592`, served through emul_scalars at
the override point, matching the predict leg's 0.16259. The only
failure is the readback's file assumption: the output dir holds the
.1.txt + input/updated yamls and NO .paramnames — an evaluate run
never writes one.**

### D-SPE2-9 (gate, CLOSED by the Architect as a declared deviation):
read the value from what the run provably produces

- Primary: parse the evaluate sampler's "Derived params:" stdout block
  (the format is in evidence, verbatim, from run 4's log) — a regex on
  OUT_NAME after the block marker.
- Secondary: the chain's own header row (cobaya's .1.txt starts with
  "# weight minuslogpost ..."), columns indexed by name directly — no
  +2 offset, weight/minuslogpost are named there too.
- The .paramnames-based readback is deleted; the D-SPE2-8 diag stays.
Probe (Mac): the shipped readback block exec'd against run 4's
VERBATIM stdout (got 0.162592, okval True — the real value passes the
5% bar at rel 4.65%) and against a synthetic header-row chain
(fallback got the same value). Runs 3 and 4 reproduced the training
numbers identically, so the path is deterministic on the workstation:
the same value returns, and the leg goes green.

### Remaining to close SPE (final)

Commit -> merge to main -> push -> workstation pull ->
`python gates/run_board.py --force-rerun scalar-smoke` -> expected
25/25 -> SPE CLOSED, CME begins. Lesson bank for CME's evaluate leg:
evaluate runs write no .paramnames; read derived values from the
stdout block or the chain header; ship the diag from day one.

## SPE CLOSED (2026-07-10, Fable) — BOARD 25/25 GREEN

**Board run 5 (HEAD d12b1ec): scalar-smoke PASS on all three legs —
val median 0.156 (bar 0.3, mean-predictor 0.455), off-center predict
rel 4.65% (bar 5%), cobaya evaluate got 0.162592 through the stdout
readback. Board summary: 0 gate(s) FAILED. The conditional close is
now UNCONDITIONAL: the SPE unit is CLOSED.**

The unit, end to end: ScalarGeometry + ScalarChi2 (zero loop change),
by-name scalar staging over one parameter file, the experiment scalar
branch (cosmolike-free by control flow), the thin driver, the
predictor scalar branch, and emul_scalars — the automatic-provides
cobaya theory the unit was commissioned for, now proven live: cobaya
asked the artifact what it provides, evaluated the emulator at an
off-center point, and returned the derived value.

The delta ledger, all closed: D-SPE1-1 (relative zero-variance guard,
both regression directions), D-SPE2-1 (sidecar uniqueness + strict
order), D-SPE2-3 (trunk-only guard), D-SPE2-4 (wrong-kind artifact
guard), D-SPE2-5 (dead-network smoke bars), D-SPE2-6 (getdist
chain-root sidecar resolution — a REAL library bug the smoke caught),
D-SPE2-7 (configs carry all six required train_args blocks), D-SPE2-8
(evaluate YAML = the proven priors+override shape; self-diagnosing
red), D-SPE2-9 (derived readback from stdout / chain header). Five
board runs total; every red was decoded from its log in one pass, and
the D-SPE2-8 diag turned the last unknown into evidence.

Standing margins, recorded: the predict bar passes at 4.65% of 5% —
thin; if a future environment shifts it past the bar, recalibrate with
that run's evidence, floor at the mean-predictor's 13.7%.

**CME (unit 2 of the pass) begins**: the spec is D-CM1..D-CM7 in
[[cmb-spectra-emulators]] under the unified handoff, plus this unit's
lesson bank (required-subscript census before the board; evaluate
readback via stdout/header; fixture sidecars in the real chain shape;
smoke bars below the mean-predictor line; ship the diag from day one).

## Scope ruling: transfer vs fine-tuning (2026-07-10, user directive)

**Transfer learning (`transfer:`) and joint refinement
(`transfer.refine`) are EXCLUSIVE to the two data-vector families —
cosmolike and CMB.** The scalar and BSN forbids are therefore
PERMANENT, not V1 deferrals: validate_scalar's transfer rejection is
the final state, and D-SP8's "unless the overlap ruling admits them"
hedge is void for transfer. **Fine-tuning (`train_args.finetune`),
by contrast, must be supported by EVERY family** — which commissions
SPE-FT below (SPE shipped with a V1 finetune forbid).

## SPE-FT — scalar fine-tuning (spec; QUEUED, small, after CME)

Lift validate_scalar's finetune forbid and route scalar runs through
the FTW warm start. The machinery is D-SP3's anticipated composition;
the work is admission + pinning + the combined driver path:

- **D-SF1 — source admissibility + the output pin.** finetune.from on
  a scalar run must rebuild a SCALAR artifact (loud wrong-kind
  otherwise, the D-SPE2-4 message pattern); the source's outputs list
  must equal the run's data.outputs EXACTLY (names and order; loud
  diff naming both lists). The source ScalarGeometry (names / center /
  scale) is PINNED into the run through build_warm_start's pinned_geom
  slot — the dv-geometry pin's exact analogue — so epoch 0 equals the
  source bitwise on shared inputs (the FTW parity contract).
- **D-SF2 — input extension.** D-FT3 applies unchanged: the target
  covmat must be a superset of the source's input names; shared
  coordinates whiten bit-identically; new columns enter zero-padded;
  the padded keys are excluded from the finetune.anchor mask (the
  existing facility — nothing new).
- **D-SF3 — the combined driver path.** validate_scalar's finetune
  rejection is REPLACED by combined validation (ia / pce / transfer /
  rescale stay forbidden; finetune admitted); from_config routes
  scalar+finetune through the warm-start branch with the D-SF1 pin;
  the WHOLE combined path forward-walked with the config-access
  census (the FTW model-block lesson — finetune YAMLs must not need a
  model: block on the scalar path either).
- **D-SF4 — gates.** scalar-identity gains the finetune legs: epoch-0
  parity bitwise vs the source on shared inputs; the outputs-mismatch
  and wrong-kind-source loud errors; anchor-with-padded-keys
  composition (assert the mask). No new board gate — the legs ride
  the existing scalar-identity.

Sequencing: after CME closes, before BSN (small); BSN then inherits
the same pattern for GridGeometry (D-BSN9 in [[baosn-emulators]]).

## SPE-FT LANDED + Mac-gated (2026-07-11, Architect, overnight mode)

- **D-SF3:** validate_scalar's finetune forbid REPLACED by admission
  (ia/pce/transfer/rescale stay forbidden); the from_config scalar
  branch gains the finetune sub-path (validate_finetune_config runs
  BEFORE any ta["model"] read — the FTW model-block lesson holds by
  ordering, probe-verified) with the D-SF1 checks: wrong-kind (a
  non-scalar source names its own family's path) and outputs-equal
  (names AND order, loud diff naming both lists).
- **D-SF1/2:** build_geometry's cosmolike finetune branch is now
  guarded `not self._scalar` too (the same latent ordering hazard
  D-CM10 fixed for cmb); the scalar branch pins the SOURCE
  ScalarGeometry wholesale (center/scale = the source standardization,
  epoch-0 parity) with the input side on extend_input_geometry (D-FT3
  unchanged; ScalarGeometry carries dest_idx/total_size so
  build_warm_start's pinned_geom slot works as-is).
- **D-SF4:** scalar_identity.py gains check_finetune: load_source
  accepts a scalar artifact (save_synthetic_scalar now stamps
  rescale="none" — a source without it is ambiguous and refused);
  epoch-0 parity via build_warm_start (same covmat, no extras);
  the extended-covmat leg asserts anchor_masks zeros EXACTLY the
  padded extra column; the outputs-mismatch and wrong-kind from_config
  legs fire before any staging (dummy data names suffice) and the
  finetune cfg carries no model: block. No new board gate — the legs
  ride scalar-identity (rerun it on the workstation).
- **D-MP5 driver pairs (scalar + cmb, commissioned by the user):**
  emulator/family_drivers.py holds the SERIAL sweep/tune loops
  single-sourced (the cosmic-shear drivers' serial paths; the
  multi-GPU pool / gpu-pack / journal machinery stays the cosmic-shear
  tool); sweep_ntrain_scalar_emulator.py / tune_scalar_emulator.py /
  sweep_ntrain_cmb_emulator.py / tune_cmb_emulator.py are thin
  namespace-ruled drivers (prog names = the D-MP5 convention; no
  --rescale — a data-vector concept each family validator rejects).
  BSN/MPS pairs land in-unit with their families.
- **Mac probe 5/5** (scratchpad probe_speft.py): compile x7; the order
  census (forbid gone, sub-path before model read, both guards, pin
  before from_targets); validate_scalar exec-probe (finetune accepted,
  transfer still loud); driver surface census (no module-level
  argparse, prog names, the experiment surface the loops call); the
  gate-leg census.
