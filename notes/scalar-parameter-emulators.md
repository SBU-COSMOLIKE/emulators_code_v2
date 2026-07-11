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
