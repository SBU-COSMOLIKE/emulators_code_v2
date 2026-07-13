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
- `rebuild_emulator(path_root, device)` (results.py): h5-driven recipe
  plus the paired `.emul` weights; v1 files refused loudly; returns
  (model, pgeom, geom, info) with info carrying ia / pce / transfer /
  family facts. "H5-only" in older prose meant "no code defaults or
  external training files", not that the weights file was optional.
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

## Red-team artifact-integrity gaps (verified 2026-07-12, open)

The resolved-recipe contract is strong, but the two physical files are
not yet one authenticated commit:

- `save_emulator` writes `<root>.emul` directly first, then opens
  `<root>.h5` directly with mode `"w"`. A history-stack, YAML, h5, disk,
  or process failure after the first write can therefore leave new
  weights beside an old, truncated, or absent record. There are no
  temporary-file commits, shared artifact id, or weights digest.
- `rebuild_emulator` validates state-dict keys and shapes only. Swapping
  two same-architecture `.emul` files loads strictly and silently uses
  the wrong weights with the surviving geometry/config. This falsifies
  any claim that a path-root pair is self-authenticating.
- Required contract: finish both temporary files before publishing;
  make the h5 the commit record for the exact weights bytes (SHA-256 +
  shared artifact id); validate that binding before `torch.load`; load
  state dicts with `weights_only=True`; and make any interrupted
  two-file publication loud rather than a mixed silent success. A
  normal Python exception before publication must leave the previous
  good pair untouched.
- Required gates: (1) swap same-shaped `.emul` files between two valid
  artifacts and require a pre-load digest failure; (2) inject an h5
  write failure after the weights temporary file exists and prove the
  old pair still rebuilds; (3) interrupt between the two final renames
  and prove the root is rejected as an incomplete commit, never loaded
  as a hybrid.

The save docstring is also incomplete: its `Arguments:` block stops at
`attrs` and omits `pce`, `pce_form`, `resolved_train`, `resolved_model`,
and `transfer_base`. The read-side `Raises:` block does not name a
missing/mismatched weights binding because none exists yet.

MPS artifacts have a second provenance gap. A grid2d h5 persists the
law name and a best-effort repository commit, but inference recomputes
the analytic Syren base from the currently checked-out source without
verifying that it is the formula used to generate the training base
dump. Vendoring prevents a package-manager upgrade from changing the
formula; it does not prevent a later repository edit from changing an
old artifact's prediction. A Syren-law artifact must persist a stable
formula digest/version and the MPS adapter must reject a mismatch before
serving. The gate changes one formula-source byte (or its declared test
digest) and proves an old artifact dies loudly; law `none` remains
unaffected.

#### Twelfth-wave extension (Architect-VERIFIED, folded into this unit): output identity — default roots collide across scientific products

Red-team finding, every load-bearing claim re-derived against code. The
run tag is not an artifact identity:

- `cosmic_shear_train_emulator.py` `run_tag` (lines 193-217) =
  `<model>[_t<T>]_ntrain<N>`: the resolved model name, an OPTIONAL
  temperature from `re.search(r"_cs_(\d+)", basename(data.train_dv))`
  (skipped when the filename has no such tag), and the staged row
  count. The save root is `cocoa_output(chains,
  f"{args.save}_{run_tag(cfg, exp)}")` (line 365) with `--save`
  defaulting to `"emulator"`; the diagnostics PDF uses the same tag
  (line 464).
- The thin family drivers all import this shared `main`
  (cmb_train_emulator.py:44, baosn_train_emulator.py:45,
  mps_train_emulator.py:47), and their example commands pass no
  `--save` — so the collisions live on the DOCUMENTED default path:
  CMB TT vs EE, BAOSN Hubble vs D_M, MPS pklin vs boost all map to
  `emulator_resmlp_ntrain<N>` (no `_cs_` tag in those filenames), and
  cosmic-shear xi vs gammat at one temperature to
  `emulator_resmlp_t<T>_ntrain<N>`. The scalar driver
  (scalar_train_emulator.py:95-114, `<model>_ntrain<N>`) collides
  across output sets the same way. Activation, model hyperparameters,
  dataset identity, split seed, NPCE, plain/fine-tune/transfer mode,
  and the source artifact are all absent from every tag.
- The overwrite is silent and total: `save_emulator` runs
  `torch.save(sd, emul_path)` (results.py:232) and
  `h5py.File(h5_path, "w")` (results.py:259) — no existence check on
  either file. A later valid run destroys a different valid emulator.
- Both run-tag docstrings state the tag exists "so runs do not
  overwrite each other" (cosmic_shear_train_emulator.py:202,
  scalar_train_emulator.py:15 and :103) — directly false across
  products; true only within one product at fixed model and N.
- The temperature tag is itself a filename regex inferring a
  scientific fact — the exact pattern the resolved-values rule (the
  standing save/load rule at the top of this note) forbids.

Adopted contract (theirs, whole): derive output identity from RESOLVED
scientific facts, never filename regexes — a readable family/product
prefix (probe, scalar output set, CMB spectrum, grid quantity, grid2d
quantity) plus a stable short digest binding the consumed
model/training configuration and the dataset-manifest identity. The
identity must distinguish plain, NPCE, fine-tune, and transfer runs,
including their source binding. Save and diagnostic products of one
run share the identity. A destination that already exists is REFUSED
before either artifact file changes, absent an explicit
overwrite/versioning action — so exact reruns cannot erase prior
evidence accidentally, and differing runs cannot alias even with
default CLI arguments. Interlock with this unit's original clauses:
the transactional two-file publication above must land WITH the
identity contract — transactionality alone would only make the wrong
replacement atomic.

Red legs (adopted): TT vs EE; Hubble vs D_M; pklin vs boost; xi vs
gammat at one temperature; two scalar output sets; two activations;
plain vs NPCE; plain vs fine-tune; two transfer sources; differing
split/dataset manifests — each pair must produce distinct roots or
fail before publication; plus an exact-root preexistence leg proving
both old files survive without an explicit overwrite.

Queue ruling (Fable): folded into THIS unit per the red team's
sequencing recommendation — pair authentication and atomic publication
are incomplete while two scientifically different runs resolve to the
same root. Priority sharpened: this unit now must land BEFORE the
five-artifact production step (the first campaign trains multiple
products of one family back to back — the exact collision path).

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

### Red-team Cobaya-contract gaps (verified 2026-07-12, open)

The artifact-only adapter tests are not a substitute for a real Cobaya
dependency-resolution test. Two current defects sit specifically at that
boundary:

- `emul_mps.get_can_support_params()` returns `Pk_grid`,
  `Pk_interpolator`, and `sigma8`. In Cobaya 3.6 this hook declares
  input parameters a component is willing to own; it does not declare
  products or derived outputs. The two P(k) products already advertise
  themselves through their `get_...` methods, while `sigma8` must be
  returned by `get_can_provide_params()`. The present mps-identity gate
  stubs `Theory`, manually assigns `output_params = []`, and therefore
  cannot exercise or falsify the real dependency routing.
- `emul_scalars.calculate()` writes `state["derived"]` only if that key
  already exists. Cobaya's calculate contract makes the component create
  the derived mapping when `want_derived` is true; the current scalar
  identity gate tests the artifact-derived name union but never calls the
  adapter's `calculate` method. A real model can therefore know that a
  scalar output is available yet receive no derived result from the
  component state.

The MPS amplitude requirement has a second naming fork. The shared Syren
reader accepts `As_1e9` or `As`, preferring `As_1e9`, but the adapter
unconditionally adds `As` to its requirements whenever either artifact
uses a Syren law. The shipped EMUL2 example samples `As_1e9` and masks the
fork by defining an extra derived `As` bridge. Requirement construction
must follow the artifact names and the shared reader's alternative-name
rule, so an `As_1e9` artifact does not require a redundant `As` parameter.

#### Eleventh-wave extension (Architect-VERIFIED, folded into this unit): adapter value-schema + multi-emulator assembly + the CMB request boundary

All five adapters whitelist extra-args KEYS but validate no VALUES —
confirmed anchors: `bool(extra_args.get("compile", False))` in every
adapter (quoted "false" enables compilation);
`os.environ.get("ROOTDIR", "")` (a missing ROOTDIR silently makes the
documented ROOTDIR-relative roots cwd-relative); `emulators` never
type-checked (a string iterates character-by-character, and a
two-character string satisfies the BAOSN/MPS len == 2 check);
`dv_return = str(get(...))` coerced not validated; the unknown-device
cpu fallback already recorded in the wave-2 entry; fast_params /
provides accept malformed iterable shapes. Contract: ONE shared
adapter-value validator used by all five classes — compile exact
bool; device an exact registered string with invalid names
distinguished from the documented unavailable-accelerator fallback;
emulators an exact nonempty sequence of unique nonempty path strings
(exactly two for BAOSN/MPS); relative paths require a defined valid
ROOTDIR; canonical roots never duplicate; provides / fast_params /
dv_return validated against their documented exact shapes BEFORE any
artifact loads.

SEPARATE composition defect (emul_cosmic_shear ~188-192, the
docstring advertises it): multi-emulator mode blindly
np.concatenate's predictions — under dv_return "3x2pt" two full
vectors become one length-2N vector; in section mode duplicate or
overlapping probe artifacts are concatenated without checking their
stored probe/section_sizes, so one likelihood block can be served
twice. Required: refuse multiple predictors in full-vector mode OR
assemble one global vector only after proving compatible layouts and
disjoint blocks; section mode requires compatible stored layouts +
unique non-overlapping probe blocks; duplicate roots/probes and a
3x2pt artifact combined with any constituent probe fail before
prediction; a valid disjoint multi-probe case keeps its defined
ordering.

CMB request boundary: must_provide applies int(lmax)
(emul_cmb ~221-224) — fractional requests truncate silently,
booleans/quoted integers accepted. Require the Cl request to be a
mapping and every lmax an exact non-bool integer in range; no
coercion.

Red legs (the red-team list adopted whole): quoted-false compile;
unknown device; missing ROOTDIR + relative root; string `emulators`
for one-root and exactly-two-root adapters; duplicate canonical
roots; malformed fast_params/provides; two full vectors; duplicate
and overlapping sections; one valid disjoint section pair;
fractional/boolean/quoted/malformed CMB lmax.

The pair validator also checks only quantity presence and exact grid
equality. It does not enforce the serve-time tuple
`pklin/Mpc3/(none|syren_linear)` plus
`boost/dimensionless/(none|syren_halofit)`. Training validates those
tuples, but the read side claims artifacts are authoritative and must
reject a malformed or hand-built h5 rather than silently treating an
unrecognized-for-that-quantity Syren law as raw output.

Required acceptance is one small real Cobaya construction (not a stubbed
base class) using synthetic artifacts: dependency resolution must assign
Pk products to the MPS theory, register sigma8 as derived, run a scalar
calculate with `want_derived=True`, and prove the returned state contains
the exact advertised outputs. Separate negative legs cover the wrong MPS
law/units tuple and an `As_1e9`-only config with no `As` bridge.

### Red-team geometry-state and covariance guards (verified 2026-07-12, open)

The persisted class marker proves which constructor to call, but neither
`rebuild_emulator` nor most `from_state` constructors validate that the
geometry tensors describe a finite invertible transform. Scalar/grid/CMB
states can carry a zero or non-finite scale; parameter and data-vector
states can carry a non-positive `sqrt_ev`, malformed basis, duplicate or
out-of-range `dest_idx`, or inconsistent center/Cinv dimensions. The model
can still load strictly because weight shapes do not authenticate those
values, then prediction returns NaN/Inf or a wrong coordinate map.

The training builders share the numerical hole. `ParamGeometry.from_covmat`,
the log-parameter builder, amplitude-factor geometry, warm-start extension,
and `DataVectorGeometry.from_cosmolike` take `sqrt(eigh(...))` without a
finite/symmetric/strictly-positive eigenvalue check. The block-diagonal
output geometry is worse: it clips every negative per-bin eigenvalue to
zero and then divides by its square root. A singular block therefore turns
an invalid covariance into infinite whitened targets instead of a loud
error; model heads repeat divisions by the inherited scales.

Required contract: one shared validation layer checks shapes, finite values,
unique/in-range indices, monotonic finite axes, positive scales/eigenvalues,
orthonormal bases within a documented tolerance, covariance symmetry and
positive definiteness, plus family registry/units tuples. Apply it on both
training construction and h5 rebuild before tensors feed a model. Do not
silently floor or clip a scientific covariance: reject it with the smallest
eigenvalue and source/bin name. Gates cover a singular covariance block, a
tiny negative eigenvalue, a zero scale in a same-shaped h5, duplicate
`dest_idx`, and a valid ill-conditioned SPD matrix just above the stated
tolerance.

## Inference numerical boundary (red-team 2026-07-12 fifth wave, Architect-VERIFIED, open; land before the EMUL2 acceptance)

EmulatorPredictor._as_row (inference.py ~442-457) checks names and
counts ONLY — its documented raises are KeyError (missing name) and
ValueError (wrong length); NaN/Inf/bool values enter the whitening
and the model, and decoded NaN/Inf returns to the caller unguarded.
CmbFactoredChi2._factor (losses/cmb.py ~316) computes exp(2 tau)/A_s
with no domain check: As = 0 -> Inf, As < 0 -> a negative factor,
tau = NaN -> NaN, none raising. Every cobaya adapter routes through
EmulatorPredictor, so this is the public inference boundary.

Contract (Implementer; the red-team block of record adopted whole):
_as_row requires every supplied input to be a real finite scalar
(bools/NaN/Inf/non-scalars rejected NAMING the stored parameter);
after pgeom.encode all encoded values finite; after the model
forward and after each branch decode, exact expected shape + finite
values before NumPy/dict conversion or scatter. Sign rules stay
family-specific: the CMB amplitude law requires finite tau, finite
strictly positive A_s, and a finite strictly positive factor (naming
the offending columns); BAOSN/MPS positivity lives in their own
queued units; NO positivity imposed on TE or generic scalar outputs.
Gates: finite control bitwise; mapping AND ordered-array NaN/Inf/
bool inputs raise naming the parameter; nonfinite encoded/model/
decoded values raise at the correct stage; As <= 0, nonfinite
As/tau, overflowed factor raise; wrong output width raises before
reshape/scatter; all five return branches covered (scalar, CMB,
grid, grid2d, data-vector).

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

## Fine-tune anchor truth (red-team 2026-07-12 eighth wave, Architect-VERIFIED, open; a training-truth unit — with or immediately after the finite/selection pair, before any anchored production fine-tune)

Two stacked defects and one live documentation lie, all verified:

1. **Config-blocked**: warmstart.py ~159-163 raises NotImplementedError
   on ANY train_args.finetune.anchor ("not implemented in V1; it lands
   as unit 2") — a STALE guard: the shared anchor facility it waited
   for exists (training.py build_anchor ~306, the decoupled anchor
   step ~288, warmstart.py anchor_masks ~887), the key is in
   _FINETUNE_KEYS (~58), and a `>= 0` validator at ~182-186 sits
   UNREACHABLE behind the rejection. README section on fine-tuning
   ADVERTISED `anchor: 1.0e-2` as available — the Architect corrected
   the README + example-YAML comment the day of the finding (docs now
   say "currently refused, restoring unit queued"); the unit restores
   the published contract.
2. **Compile no-op behind it**: once the stale guard is removed, the
   reference state and masks carry EAGER names (the source is rebuilt
   "never torch.compile'd", warmstart.py ~86/271) while the live model
   is compiled and exposes `_orig_mod.`-prefixed names; build_anchor
   silently skips any parameter absent from the reference (by design
   for frozen params, ~309-311), so on the production CUDA default the
   anchor matches ZERO parameters, Anchor.entries is empty, and the
   artifact records an anchor value that never ran. Transfer-refine's
   base anchor appears eager (not automatically the same failure) but
   must be gate-tested.

Contract (the two red-team blocks of record adopted whole): remove
the obsolete rejection; the finite nonnegative anchor validator
becomes live (NaN/Inf lambda raises — NaN passes >= 0 today);
canonicalize compile prefixes at ONE boundary (or anchor the
underlying eager module) with exact one-to-one coverage; a positive
anchor with zero matched trainable parameters FAILS; report/validate
matched, masked, frozen, and unexpected names; masks go through the
identical canonical mapping; persist an EXECUTED anchor record
(matched count + effective lambda), never configuration alone. Gates:
a real from_config leg proving a positive anchor reaches training
(not just helper tests); the same one-step known-answer update eager
vs compiled, both moving by the exact anchor formula and matching;
padded masks proving extra columns stay free; missing/extra reference
names; lambda-zero identity; transfer-refine eager coverage; nonzero
matched count asserted for a positive anchor; the resolved artifact
record inspected.

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

### transfer-identity cross-family leg: FIXTURE DEFECT (board run 12, root-caused; fix pending)

The run-12 red (the only board red, 58/59 legs green) is in the GATE
FIXTURE, not the library. The "cross-family transfer base raises" leg
(gates/checks/transfer_identity.py, the end of the diagonal section)
builds a grid2d config whose `transfer.from` points at the
`diag_transfer` artifact saved two legs earlier — an artifact that
EMBEDS a transfer_base group (it exists to test composed prediction).
`_load_diag_transfer` (emulator/experiment.py) calls
`warmstart.load_source` BEFORE its family-kind check, and load_source's
chaining refusal ("chaining a transfer over a transfer is out of scope
(no chaining)") fires first; the leg's needle test wants "never" and
"families", so it fails on the wrong — but equally correct — message.
The library's cross-family rule is implemented and correctly ordered
(kind check immediately after load; message "a transfer never crosses
families"); the fixture hands it an artifact invalid in TWO ways, and
the other guard answers first.

**Fix spec (Implementer; gate file only, library frozen):** in that
leg, save a PLAIN grid base artifact (the leg's local `base` net +
`geom` through the same save_emulator call, WITHOUT the transfer_base=
argument, with the same rescale="none" attrs) at its own root, and
point the grid2d config's `transfer.from` at THAT root. load_source
then succeeds, the kind check sees GridGeometry != Grid2DGeometry, and
the cross-family ValueError fires with both needles. No other leg
changes; the chaining refusal keeps its own dedicated green leg
(lifecycle: "chaining refused"). Lesson (also in gates-and-board.md
run 12): a loud-error leg's fixture must be invalid ONLY in the way
under test.

The same Implementer unit carries the gate METADATA truth-up the red
team tied to the five-leg close (their 2026-07-12 acceptance): the
gates/board.py cmb-identity registry entry (its `maps` field and the
gate docstring) must name all five covariance legs — the exact
contraction, the old-weight miss, raw-vs-scaled fixture integrity, the
width-3 band projection, the exact zero-band weight — in plain
language ("independent known-answer check", never "oracle" in prose;
the `check_covariance_oracle` identifier itself stays). The three
READMEs were already trued up by the Architect (run-12 status +
five-leg rows).

#### Fixture-fix resume (2026-07-12, Opus) — awaiting Architect audit

Built, gate files only (`emulator/` frozen). Two parts:

- Part 1, `gates/checks/transfer_identity.py` `check_diagonal`: the
  cross-family leg now saves a PLAIN grid base — this leg's own `base`
  net + `geom` (a `GridGeometry`) through the same `save_emulator` call,
  no `transfer_base=`, same `rescale="none"` / `quantity="Hubble"`
  attrs, `resolved_model=grid_base_recipe(names, int(z.size))` — at its
  own root `plain_grid_base`, and points the grid2d config's
  `transfer.from` there (was the `diag_transfer` root, which embeds a
  transfer_base group). Static trace of the frozen library path
  (`_load_diag_transfer`, experiment.py:1410-1426): `transfer` present ->
  `validate_transfer(diagonal=True)` accepts `form: sum` ->
  `warmstart.load_source(plain_root)` SUCCEEDS (no transfer_base group,
  so the chaining refusal at warmstart.py:340 never fires) ->
  `got_cls = "GridGeometry"` != the grid2d branch's
  `geom_cls_name="Grid2DGeometry"` (experiment.py:2327) -> raises
  "a transfer never crosses families", so both needles (`"never"`,
  `"families"`) pass. The chaining refusal keeps its own dedicated leg in
  `check_lifecycle` ("chaining refused"). The leg needs torch
  (save_emulator / from_config), so this is a static trace, not a Mac
  exec run.
- Part 2, `gates/board.py` cmb-identity entry: the `maps` field, the
  `gate_cme_a` docstring, and the board `label` now name all five
  covariance legs in plain language — the exact contraction, the
  old-weight miss, the raw-vs-scaled fixture integrity, the width-3 band
  projection, and the exact zero-band weight — as an "independent
  known-answer" check; "oracle" no longer appears in prose (the
  `check_covariance_oracle` identifier reference stays).

Mac gate: `py_compile` clean on both files (`transfer_identity.py`,
`board.py`); the fixed leg's correctness is the static trace above (no
torch on this box). No `emulator/` change (library frozen); cmb-smoke
needs no rerun (gate-file-only, producer unchanged).

Close (user-run, workstation, after the merge): `python
gates/run_board.py --force-rerun cmb-identity transfer-identity` —
transfer-identity proves the fixture fix, cmb-identity re-executes the
five-leg delta and its board text must match the five executed legs;
both green closes the run-12 red.

#### Fixture-fix Architect audit (2026-07-12, Fable): ACCEPTED, committed

Audited against the raw diff of both files. Part 1 matches the spec
verbatim (the plain grid base is invalid ONLY in the way under test;
the chaining refusal keeps its dedicated leg); the static trace
matches the Architect's own run-12 root-cause chain independently
derived from the same code. Part 2 verified line by line: the
docstring, `maps`, and expect label name the five legs plainly, the
raw-vs-scaled boundary description is physically accurate, and the
only "oracle" left in board.py is the kept identifier (grep-proven,
untruncated). py_compile re-run by the Architect: OK. Committed on
the branch by the Architect with Implementer attribution; the grid2d
unit's files stay uncommitted pending its REVISION (the stable-moments
amendment — see data-generation-and-cuts.md). Close = the two-gate
force-rerun above.

## Follow-the-IDs (git archaeology)

FTW: D-FT1..10, D-FTW-1/2. TPE: D-TP1..10, D-TPE-1, D-TPE2-1..3,
Ruling 1 (reassociation). Schema: D-SV1/2, D-CT1..3, GCT-D
(dv_return), Riders 1-5 (paths, .paramnames cross-check, GitHub math
policy). GEO: D-GEO1..5. Board homes here: save-rebuild-drift,
cobaya-adapter, finetune-identity/smoke, transfer-identity/smoke,
geo-paths.

## Artifacts are not bound to the code that gives their weights meaning (red-team 45M-13, 2026-07-12, Architect-VERIFIED; queue 37 — folds into the artifact-integrity campaign beside unit 3, implementation identity distinct from pair identity)

save_emulator writes git_commit as best-effort provenance
(results.py:398, "unknown" fallback :403) and its documentation
claims it "marks a v2 file (rebuild refuses one without)" (:192) —
but rebuild_emulator never reads or validates git_commit (full grep:
three hits, none in rebuild; the only v2 test is schema_version == 2
plus the recipe). The recipe stores class NAMES and constructor
values, then imports the CURRENT class/activation/normalization/
geometry/decoder/transfer/PCE implementation — a later code change
under the same names is silently accepted. Sharpest consequence, the
syren-law MPS path: training learns r = log(P / P_base_old) but
inference recomputes the base from the current syren/ source, so a
base change serves P * (P_base_new / P_base_old) with every weight
loading strictly — vendoring stops a pip upgrade, it does not bind an
old artifact to the vendored implementation that generated its
targets.

Contract (Implementer): (1) a compatibility manifest persisted for
every implementation whose behavior interprets the weights
(model/design, activation, normalization, parameter/output geometry,
decoder/loss composition, any analytic base such as syren);
(2) explicit semantic implementation identifiers or content hashes
backed by a versioned registry — a raw git commit stays provenance,
never the compatibility mechanism; (3) rebuild compares every
required identifier BEFORE importing/serving; a mismatch selects a
retained versioned implementation or refuses with the artifact value,
runtime value, and migration/retraining action; (4) the MPS artifact
bound to the exact syren base variant that generated its law-space
targets (with 45M-14's base-variant naming, queue 38); (5) the
results.py git_commit documentation corrected — enforce the claim or
describe it as unvalidated provenance; (6) folded into the
artifact-integrity campaign, but pair identity does NOT substitute
for implementation identity. Gate (torch/HDF5, board): save under id
A rebuilds under A; changing only the syren/base id refuses before
prediction; changing only a model/geometry semantic id refuses; an
unrelated repo commit with unchanged semantic ids still rebuilds; a
registered migration selects the old implementation and reproduces a
stored known-answer prediction; a deleted manifest key fails loudly.

## config_resolved_yaml does not record what the run consumed (red-team 45M-19, 2026-07-12, Architect-VERIFIED; queue 41)

run_emulator's resolved_train labels itself "defaults materialized"
but stores raw inputs: training.py:3028 records `"loss": loss` (the
user's block — null when absent, though training consumed the default
sqrt mode; omitted BerHu knots absent though materialized values were
consumed); each phase computes effective loss/berhu/ema/trim/focus/
clip/rewind/lr/wmupe/sched passes but the artifact stores raw
trunk/head overlays and only the top-level scheduler; transfer
refinement reuses the final pass's effective settings but persists
only {epochs, base_lr_scale, anchor}; histories include refinement
epochs while nepochs stays pre-refinement. Reconstruction therefore
depends on TODAY'S default-resolution code — the exact drift channel
the resolved record exists to remove (the house never-trust-defaults
doctrine, violated on its own flagship surface).

Contract (Implementer): (1) persist a `passes` sequence in execution
order — phase name, model train phase, epoch count, computed lr,
warmup, scheduler class + resolved kwargs, fully resolved loss, trim,
focus, clipping, rewind, EMA; (2) transfer refinement gets its own
pass entry with every inherited effective value; (3) run-level
roughness persisted separately if it configures the loss object once;
(4) total_epochs persisted and required to equal every history
dataset's row count; (5) the raw YAML kept separately as provenance,
never overwritten by the resolved form; (6) loading/reporting reads
the resolved pass record, never re-runs inheritance logic. Acceptance
(short torch training/artifact gate; config-only probes pure): absent
loss block records the consumed sqrt mode, not null; omitted BerHu
knots appear with consumed numerics; trunk/head overrides produce two
complete pass records; ema null in one phase recorded as disabled
there; refinement is a third complete record; history lengths
inconsistent with total_epochs fail before publication.

### 45M-08 amendment to the unit-3 campaign (index-received; Architect-verified directly)

The full 45M-08 block was not relayed, but its finding is verified
from the code already in evidence: compute_cmb_covariance.py:766-768
publishes with an unconditional `np.savez(out_path, **out)` — no
preexistence refusal, no temp-file + rename, so a rerun overwrites a
prior covariance artifact and a mid-write death leaves a partial
file. This is exactly unit 3's transactional-publication +
preexistence-refusal contract extended to the covariance producer;
adopted as a clause of unit 3 (the twelfth-wave extension above), not
a new unit. Gate leg: rerun-with-existing-output refuses; kill-mid-
publication leaves the prior artifact intact.

### 45M-33 amendment to unit 41 (transfer-refine artifact truth): the drift metric is buffer-diluted and lies at zero reference (2026-07-12, Architect-VERIFIED)

The refined-transfer driver advertises relative WEIGHT drift but
iterates state_dict() (cosmic_shear_train_emulator.py:421) —
persistent non-trainable buffers (ResCNN/ResTRF pad_idx layout
indices; PCE fixed buffers) contribute zero numerator but their
squared values inflate sq_den, so the advertised total is diluted by
layout metadata that cannot drift. Second false report at :427:
`rel = sqrt(dn/d0) if d0 > 0.0 else 0.0` — a pretrained tensor that
is exactly zero and MOVES records drift 0.0 (zero reference norm
makes relative drift undefined, not zero). Both full states are
retained so prediction is unaffected; the defect is a reassuring
false summary in root metadata. Amendment: the metric is defined over
TRAINABLE parameters only via an explicit canonical key set
(buffers/layout/state excluded); numerator and reference norms (or
absolute drift + status) persisted beside any relative value;
||W0|| == 0 reports exact zero ONLY when the drift norm is also zero,
else a named zero-reference/nonzero-absolute-drift status — never
relative 0.0; parameter-key equality between the two states verified
before publication; the stored summary exactly recomputable from the
persisted states + declared key set; prose/attrs renamed if the
chosen metric is state drift. Red legs (torch, board-listed): a huge
fixed integer buffer + one moved parameter — total equals the hand
calculation, invariant to buffer magnitude; zero-reference unchanged
-> exact zero; zero-reference moved -> absolute drift + status, never
0; missing/extra key raises before publication; multi-parameter known
answer; readback recomputes the summary.

## UNIT 7 AMENDED (45M-45): syren parameter aliases can disagree silently — the alias-consistency boundary

Adjudicated + reproduced LIVE (Fable, 2026-07-12; the vendored syren
runs on the Mac). syren_params_from (emulator/syren_base.py) silently
prefers As_1e9 over As (:67-70) and w over w0 (:82-87); the discarded
name is never compared against the chosen one. Reachable through BOTH
public callers, each of which passes a complete resolved mapping: the
generator hands the full Cobaya to_input dict to the reader
(dataset_generator_mps.py, the write_base block), and the adapter
hands its full calculate(**params) mapping (emul_mps.py). The shipped
EMUL2 evaluate YAML DEFINES BOTH amplitude names to bridge the
requirement fork (README ~:2214-2216), so "both aliases present" is
an exercised shipped shape, not an invented internal call.
Reproduced: {As_1e9: 2.1, As: 9e-9} -> the resolver takes 2.1 and
exactly ignores As; base_pklin max relative difference 0.7667 against
the As value's base. {w: -1.0, w0: -0.7} -> takes -1.0; max rel
0.2449 (the red team's grid gave 0.2403 — same mechanism, the
magnitude is grid-dependent). ALL probe spectra finite and positive,
so the downstream MPS positivity/finiteness guards cannot catch the
conflict. Worst case is the law-correction artifact: the network half
evaluates from the artifact's stored parameter names while the
analytic base follows this independent precedence rule — the two
halves can describe different cosmologies with no exception raised.

Contract (folds into THIS unit's adapter/ingress campaign — no second
syren implementation):
1. syren_params_from remains the ONE shared generator/adapter
   authority; alias pairs become an explicit consistency boundary.
2. Amplitude: As_1e9 alone or As alone is accepted. BOTH present =>
   require As_1e9 == 1e9 * As within a documented
   float-representation tolerance; otherwise raise naming both values
   and the conversion.
3. Dark energy: w alone or w0 alone. BOTH present => numerical
   equality within the same stated representation policy; otherwise
   raise naming both.
4. Canonicalize only AFTER the consistency proof — never silently
   prefer one conflicting value.
5. LCDM rule preserved: neither w nor w0 present => canonical
   w0 = -1 (a model fact, not a config default); absent wa likewise.
6. No duplication of the queued value-schema work: finite non-bool
   real validation stays with generator ingress / adapter value
   validation; THIS amendment owns only the cross-alias equality that
   individual-value validation cannot establish.
7. Requirement construction (this unit's existing As/As_1e9 routing
   clause) must not REQUIRE a redundant amplitude alias; a second
   alias supplied deliberately for another component is verified, not
   ignored.
8. On failure: emul_mps.calculate leaves NO Pk_grid, interpolator, or
   derived state keys; the generator rejects the sample BEFORE
   writing any raw/base payload row.
9. Documentation defines the aliases definitionally (As_1e9 = 10^9
   As; w and w0 are two names for one present-day equation-of-state
   value); "prefers" wording is removed as a correctness policy.

Red legs: As_1e9-only and As-only controls resolve to the same
canonical amplitude and reproduce current base values; consistent
dual amplitude names pass; inconsistent dual amplitude names raise
BEFORE base_pklin or base_boost; w-only and w0-only controls
reproduce current values; consistent dual dark-energy names pass,
inconsistent names raise; mutation arm — restore the current
first-alias-wins branch; both conflict legs must fail the gate;
adapter state-integrity leg — an inconsistent dual alias leaves no
partial Pk state; generator leg — the same inconsistency is rejected
before _dv_write, no base/raw quantity published for the row;
shipped-bridge control — the actual As_1e9 -> As bridge configuration
remains accepted. The resolver/base probes are numpy-only
(Mac-runnable); the adapter/generator integration legs use the
torch-backed fixtures under gates/checks/, LISTED on the board,
workstation-run.

## UNIT 11 AMENDED (45M-49): the float32-resolution guard can itself store a zero divisor

Sixth 45M batch (2026-07-12), Architect-verified and reproduced on
Mac before placement. The un-standardizable-column guard is a
float64 RELATIVE test, but the constructor stores float32: the guard
can accept a column whose stored scale is exactly zero.

Verified sites (current HEAD): ScalarGeometry.from_targets computes
center/scale in float64 and refuses only on
scale <= 8 * eps32 * |center| (scalar.py:145-146); the constructor
then casts both to float32 unconditionally (scalar.py:85-90) while
its own docstring promises a strictly positive scale.
GridGeometry.from_targets repeats the sequence (grid.py:189-192).
Grid2DGeometry.from_stats likewise (grid2d.py:196-197), and grid2d's
const_mask is decided by the SAME float64 relative rule — a column
that underflows on storage is not entered in const_mask, so it is
neither pinned nor refused (grid2d.py:198-223): the stored scale is
0.0 and the point is unusable.

Reproduction (numpy, Mac): f = nextafter(float32(0), float32(1)) =
1.401298464324817e-45, the smallest float32 subnormal — representable
in the float32 dumps from_targets reads. targets [0, 0, f]: float64
center 4.670995e-46, population std 6.605784e-46, guard threshold
4.454608e-52 -> ACCEPTED; stored float32 center 0.0, stored float32
scale 0.0; encode divides by exact zero (probe returns inf).
targets [0, f, f]: accepted; stored center nonzero (f survives as
the smallest subnormal) with stored scale 0.0 — the same zero
divisor behind a nonzero center. The relative rule cannot see
absolute underflow: both float64 moments sit below float32's
subnormal floor while their ratio looks healthy.

Contract (amends unit 11; the from_state read-side clause and both
covariance sqrt mechanisms recorded above stand unchanged):

1. Validate the values that will actually be STORED, never only the
   pre-cast estimates: cast every produced center/scale pair through
   the declared artifact/model dtype (float32 today) first.
2. Require a finite stored center and a finite, STRICTLY POSITIVE
   stored scale for every pair; refusal names the column or grid
   coordinate exactly as the current errors do.
3. Apply the relative-resolution rule IN that same storage
   representation, so the two tests compose: absolute underflow and
   relative collapse are both refused.
4. Grid2d classifies a post-cast-zero scale as CONSTANT before the
   partial-pin vs whole-surface-refusal decision — the pin and
   dead-dump policies operate on stored-representation facts.
5. The same representability rule applies to this unit's covariance
   square-root scales (mechanisms (a) and (b) above): a positive
   float64 eigenvalue is not sufficient if its stored float32 square
   root is zero; refusal names the smallest stored scale.

Red legs: scalar [0, 0, f] is refused before construction; scalar
[0, f, f] is refused; one grid column with that payload is refused
naming its coordinate; one grid2d column with that payload is pinned
consistently while an all-such surface is refused as dead; an
otherwise valid scale just above the stated stored-dtype boundary
survives and round-trips (state -> from_state bitwise); a covariance
eigen-scale that underflows on storage is refused naming the
smallest stored scale. The constructor/encode legs are torch legs:
they join the existing geometry identity checks under gates/checks/
(scalar-identity, bsn-identity, mps-identity own their geometries),
LISTED on the board, run on the torch workstation. The arithmetic
discrimination above stays a numpy-only companion leg (Mac-runnable).

## UNIT 24 AMENDED (45M-50): scalar fine-tuning erases its source provenance at save

Seventh 45M batch (2026-07-12), Architect-verified on HEAD. Public
reachability confirmed: validate_scalar states "fine-tuning IS
supported"; the from_config scalar fine-tune branch loads the
source, pins outputs, and sets _scalar/_finetune/_finetune_root
(experiment.py ~2055-2069). This note's universal fine-tune contract
(the "Provenance attrs: finetuned_from + finetune_extra_names" line)
claims every family. The shared family driver honors it
(cosmic_shear_train_emulator.py:385-387); the scalar driver's own
save path does not — its attrs mapping (scalar_train_emulator.py
:201-211) records model/data/best metrics only and never inspects
exp._finetune. Untruncated grep: `finetuned_from` is written by the
shared driver ALONE (the only other hits are the finetune-smoke gate
asserting it — a gate the scalar path never crosses). A scalar
fine-tuned artifact is therefore valid and inference-loadable with
no declaration that its initialization and architecture came from
another artifact; cold and fine-tuned runs produce the same
provenance key set.

Contract (amends unit 24 — the anchor-truth clauses stand):

1. One shared artifact-provenance assembler owns the common attrs
   for EVERY training driver; scalar adds its family facts (outputs,
   train/val parameter filenames) but never forks
   fine-tune/anchor/provenance logic.
2. Cold run: no fine-tune attrs. Fine-tune run: canonical resolved
   source identity plus ordered extra names.
3. The recorded source identity is the resolved root/digest actually
   loaded, never the raw YAML spelling.
4. Once unit 24 opens anchors, the executed-anchor record shares
   this same path — not added to only the shared cosmic driver.

Red legs: scalar cold save omits fine-tune attrs; scalar
names-equal fine-tune saves finetuned_from and an explicit EMPTY
extra-name list; scalar extended-input fine-tune preserves ordered
extra names; scalar and one shared-driver family produce the same
common provenance schema; mutation arm — delete the scalar
provenance branch / shared-assembler call and the artifact readback
gate must fail. Save/readback legs use torch to build the synthetic
artifacts: they live under the existing board-listed
finetune-identity check in gates/checks/, run by the user on the
torch workstation.

## UNIT 11 AMENDED (45M-65, fifteenth batch, 2026-07-12): parameter-side covariance ingestion joins the read-side integrity contract

Finding (red team, CONFIRMED live): validate_scalar imposes no
minimum parameter count (any nonempty input set is legal), and
the ordinary scalar build then calls ParamGeometry.from_covmat
(experiment.py:3488-3491), which is loadtxt -> eigh with ZERO
validation (parameter.py:128-132). A scientifically valid
one-parameter covariance file ("# x" then "4.0") loads as a
0-DIMENSIONAL array — np.loadtxt returns shape (), not (1, 1) —
and np.linalg.eigh deterministically raises
LinAlgError("0-dimensional array given...") before training
starts (proven through the REAL from_covmat, 2026-07-12). The
same probe proved the multi-parameter path is a LIVE instance of
this unit's mechanism (a) on the parameter side: a covmat with a
negative variance flows loadtxt -> eigh -> np.sqrt silently to
sqrt_ev = [nan, 1.0] — the NaN whitening scale built without a
word. The dimensional collapse repeats on the sample-derived
path (np.cov(one-feature, rowvar=False) is also 0-d,
parameter.py:264) and the unvalidated loadtxt -> eigh shape
repeats at AmplitudeFactorGeometry.from_covmat (:422-437 — a
THIRD site, found in adjudication, which the finding did not
cite; note np.ix_ on a 0-d cov breaks there before eigh would).
Distinct from the queued one-ROW parameter-table fix: that is a
dataset-shape defect; this is a valid one-PARAMETER geometry
refused by dimensional accident.

Contract (the red team's six clauses adopted; folded into this
unit's single mechanism):

1. Covariance input is normalized to an exact two-dimensional
   square matrix before eigendecomposition (a valid scalar
   covariance becomes (1, 1)); normalization NEVER blesses
   malformed input — it feeds the same integrity checks.
2. Header-name count, covariance width, and center width must
   agree exactly; a mismatch is refused naming all three observed
   dimensions.
3. Finite, symmetric, strictly positive-definite values under
   this unit's geometry-integrity policy at ALL parameter-side
   sites — the nan-sqrt_ev probe result makes parameter.py a
   proven instance of mechanism (a), not a precaution.
4. The dimensional-totality rule applies wherever covariance is
   derived from samples (np.cov, one feature) and at
   AmplitudeFactorGeometry.from_covmat (in scope).
5. All multi-parameter numerics byte-for-byte.
6. No new mechanism: this amends unit 11; the validation home is
   shared with the output.py covariance sites already recorded
   here.

Red legs (CPU numpy legs; the round-trip legs are torch and live
in the board-listed scalar-identity gate — Architect placement,
per the red team's request):

- one header name + positive 1x1 covariance builds and
  round-trips decode(encode(x)) == x;
- the whitened displacement has the analytic value
  (x - center)/sqrt(var);
- zero, negative, NaN, and Inf scalar variances are refused
  BEFORE eigh/sqrt;
- header count, center width, and covariance dimension mismatches
  are refused with the three observed dimensions named;
- a normal multi-parameter control remains byte-identical;
- the sample-derived one-feature constructor, while public API,
  receives the same one-dimensional acceptance test.

Placement: unit 11, where queued (no reshuffle). USER-VISIBLE:
one-parameter emulators become buildable (today they crash before
training); malformed covariances are refused loudly (today a
negative variance trains with NaN whitening).

## Structured evidence map — gate contract anchors (45M-72 foundation)

The board's structured evidence map (`Gate.evidence`) pins each migrated
gate to a stable, runner-validated anchor in its home note; the mechanism
and the audited rollout are documented in `gates-and-board.md`. The
artifact-side gate anchors here:

<a id="arb-a-artifact-readback"></a>
**artifact-readback (ARB-A) — saved attributes are parsed by type, not
truthiness.** The shared typed reader accepts a native boolean, returns the
default for an absent key, and refuses every string / integer (the truthy
`"False"` that would otherwise load drifted transfer weights) naming the
file and the schema; a source census confirms no artifact boolean is
truthiness-coerced. The live save / forge / rebuild proof is
workstation-owed.
