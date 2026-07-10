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
