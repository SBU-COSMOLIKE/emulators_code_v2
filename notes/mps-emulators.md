# Matter power spectrum emulators + hybrid inference (spec)

**Date:** 2026-07-11. **Status:** SPEC (Architect) — QUEUED as a major
unit after BSN (it consumes BSN's GridGeometry pattern + the D-CM3-A
generator core). **Spec code:** MPS. Home note for the gates
`mps-identity` / `mps-smoke`. Assigned by the user "all that is you to
implement" (Architect implements, the overnight-mode precedent).

## The request (user, 2026-07-11)

Port the legacy MPS emulator (Downloads/emulators_code-main/emulmps/,
580-line cobaya Theory over keras models; structure read):
1. **It corrects an approximate formula** — the legacy base is the
   symbolic_pofk syren-halofit family; the true emulator learns the
   CORRECTION to that formula (the base+refiner philosophy with an
   ANALYTIC physics base).
2. **TensorFlow -> PyTorch**: rewrite in the v2 stack; the .keras +
   metadata.joblib artifacts stay on the legacy classes
   (replaced-not-ported, D-SP6 — RETRAIN via the new generator).
3. **The grids are fixed** (the legacy generator, read): z = concat of
   linspace(0,2,100,endpoint=False) + linspace(2,10,10,endpoint=False)
   + linspace(10,50,12) = 122 redshifts; k = logspace(-4, 2, 2000).
   TWO quantities: linear P(k,z), and the nonlinear BOOST
   B(k,z) = P_nl/P_lin (the ratio is the emulated target).
4. **dataset_generator_mps.py** is primitive (ad-hoc MPI, LHS+uniform,
   no checkpoints) — rewrite it to the D-CM3-A generator-core level as
   the FOURTH thin driver (no duplication).
5. **EMUL2 = hybrid inference** (EXAMPLE_EMUL2_EVALUATE1.yaml, saved
   by the user): the cosmolike likelihood runs with use_emulator: 2,
   consuming EMULATED CAMB PRODUCTS — Pk from emul_mps, distances/H
   from emul_baosn, rdrag from the SPE emulator — instead of a full
   data-vector emulator. The v2 emul_mps must provide the
   Pk_interpolator products that mode consumes.

## Design rules

- **D-MP1 — outputs = two grid functions.** A 2D GridGeometry over
  (z, k) — the BSN GridGeometry generalized to two stored axes,
  flattened row-major for the network; grids/units/law persisted
  in-artifact (never the legacy sidecar bundles). Two artifacts:
  "pklin" and "boost". Output widths are large (122 x 2000 raw) —
  V1 trains on a DOWNSAMPLED k grid (a knob, persisted; the adapter
  interpolates, exactly as CAMB's own interpolator does); PCA output
  compression stays the recorded evidence-gated V2 (the phiphi
  ruling).
- **D-MP2 — the imposed formula base (the headline).** The target is
  the correction to the analytic base: target = log(P / P_base(k,z,θ))
  for pklin (base = the syren linear formula) and log(B / B_base) for
  the boost (base = syren-halofit), through the LAW-REGISTRY pattern
  (D-CM2/BSN): law names "syren_linear" / "syren_halofit" / "none",
  persisted by name; decode multiplies the base back. The base
  formulas come from the symbolic_pofk package (a dependency the
  legacy already carries); READ legacy emulmps.py lines 27-66 before
  implementing (the import fallbacks + run_halofit conventions are
  binding).
- **D-MP3 — the generator.** dataset_generator_mps.py, the fourth
  thin driver on generator_core: per sample one CAMB-in-cobaya call
  through the Pk_interpolator requirements (BINDING legacy quirk,
  keep the comment: a Cl requirement must ride along — "DONT REMOVE
  THIS - SOME WEIRD BEHAVIOR IN CAMB WITHOUT WANTS_CL"); writes TWO dv
  files from one pass (pklin rows + boost rows on the persisted
  grids); params/.paramnames/.covmat from the core. extrap_kmax=200
  and the mead2020 halofit choice ride the training YAML, persisted.
- **D-MP4 — the cobaya adapter (hybrid mode).** cobaya_theory/
  emul_mps.py on the emul_scalars template: artifact-derived
  requirements; provides Pk_interpolator (linear + nonlinear via
  P_nl = P_lin * B) through a CAMB-compatible RectBivariateSpline
  interpolator — PORT the legacy PowerSpectrumInterpolator class
  (lines 68-198, itself adapted from CAMB/Antony Lewis; keep the
  attribution). Must satisfy the EMUL2 YAML's consumer set: the
  cosmolike likelihood at use_emulator: 2 + emul_baosn + the rdrag
  emulator, all in one run. The wrong-kind guards both ways.
- **D-MP5 — drivers + THE NAMESPACE RULING (applies program-wide).**
  New drivers follow `<verb>_<family>_emulator.py`
  (train_/tune_/sweep_ntrain_ x cmb/baosn/mps/scalar); commissioned
  NOW by the user: chi2-vs-N_train sweep drivers AND optuna tune
  drivers for SN/BAO (baosn), CMB, and MPS — each landing WITH or
  right after its family unit, built as thin wrappers over the
  existing sweep_ntrain/tune machinery (family = the from_config
  branch, so the wrappers are mostly config plumbing). The README
  gains a "Drivers" subsection with a table: file, family, verb, what
  it produces (the user: "some prefix and suffix so it is clear this
  is about CMB or MPS"). Renaming the EXISTING cosmic-shear drivers
  = a POL-1 item (board configs + README references move with it),
  recorded there.
- **D-MP6 — gates.** mps-identity (grid geometry 2D round-trip +
  law-exactness on synthetic formulas + save/rebuild/predict bitwise
  + wrong-kind/error legs, torch-only via the stub pattern);
  mps-smoke (tiny generated dump through real CAMB, 2-epoch train of
  the boost, a cobaya evaluate through emul_mps in a MINIMAL hybrid
  YAML — the full EMUL2 cosmolike run is the acceptance experiment,
  not the smoke). The full SPE lesson bank binds (subscript census,
  dead-network bars off-fiducial, priors+override evaluate, stdout
  readback, diag from day one).
- **D-MP7 — out of scope (recorded).** PCA output heads (V2,
  evidence-gated); transfer (PERMANENT forbid — the scope ruling:
  transfer = cosmolike + CMB dv only); the legacy .keras weights
  (retrain); syren-formula refits. Fine-tuning IS in scope (the
  universal rule) via the grid-geometry pin (the SPE-FT/BSN pattern).

## Sequencing (the full queue after this commission)

CME (increments 3-5 remain) -> SPE-FT (small) -> BSN -> MPS (this) ->
GEO -> POL. The per-family tune/sweep drivers (D-MP5) land with their
families: CME's pair right after CME closes, BSN's and MPS's in-unit;
the scalar pair rides SPE-FT. Hybrid-inference acceptance (the EMUL2
YAML end to end) closes the MPS unit. Board: +2 gates per family.

## Links

[[cmb-spectra-emulators]] (law registry, D-CM3-A core, D-CM11),
[[baosn-emulators]] (GridGeometry), [[scalar-parameter-emulators]]
(adapter template + lesson bank), [[post-program-polish]] (driver
renames), [[geometry-family-folder]].

## D-MP2-A — where the syren base lives (2026-07-11, Architect ruling)

The syren base is UNLIKE every earlier law: log_offset is a constant,
as_exp2tau a per-row scalar — but P_base(k, z, theta) is a per-row
FUNCTION of the cosmology, computed by numpy formulas (symbolic_pofk).
Putting it inside the geometry's encode/decode would drag numpy into
the compiled training loop (a GPU sync per batch) or force a torch
port of the syren formulas (a drift channel against the very formula
the emulator corrects). RULING:

1. **The generator writes the base beside the raw dump.** Per sample,
   dataset_generator_mps.py computes P_lin / boost from CAMB AND the
   syren base from symbolic_pofk (both numpy-land), writing
   {dvsf}_pklin.npy + {dvsf}_pklin_base.npy (and _boost /
   _boost_base), plus the _z.npy/_k.npy grid sidecars. Resolved
   values: the base a training consumed is ON DISK, never recomputed
   under a possibly-drifted package version.
2. **Staging forms the law-space target once (cold):**
   target = log(P / P_base) row by row at staging; the 2D
   GridGeometry standardizes law-space rows (torch-pure encode/
   decode = standardize/destandardize only). The law NAME + the
   downsample spec persist in the artifact.
3. **One base module:** emulator/syren_base.py wraps the
   symbolic_pofk calls (the legacy emulmps.py lines 27-66 import
   fallbacks + As_to_sigma8 -> plin_emulated -> run_halofit
   conventions, moved verbatim) — used by the generator, by emul_mps
   (decode: P = exp(destd) * base(theta), the legacy get_pks
   use_syren=True flow), and by the gates. The base has exactly one
   definition; the artifact's law name selects it.
4. **The predictor's mps branch returns the LAW-SPACE grid**
   ({"z", "k", "log_ratio"}); emul_mps (and any profile script)
   multiplies the base back through syren_base — the one documented
   consumer-side step, mirrored on the README's two-door pattern.

## Resume state (Architect implementing directly, overnight mode)

**2026-07-11: BSN closed (board 29); MPS increment 1 IN PROGRESS.**
Binding legacy facts captured this window (all read in full):

- PowerSpectrumInterpolator = legacy emulmps.py lines 68-196 (adapted
  from CAMB / Antony Lewis, attribution kept): RectBivariateSpline
  over (z, log k), logP mode with logsign, extrap_kmin/kmax power-law
  pads (two extra columns each side), check_ranges loud, P()/logP(),
  __call__ warn. Port VERBATIM into cobaya_theory/emul_mps.py (its
  LoggedError/get_logger imports ride cobaya, fine there).
- The legacy adapter surface to keep: get_Pk_grid (delta_tot only,
  loud otherwise; state keys ("Pk_grid", nonlinear, "delta_tot",
  "delta_tot")), get_Pk_interpolator (log_p detection with the
  zero-crossing fallback + the state cache key incl. extrap bounds),
  _compute_sigma8 (R=8 tophat over log k via trapz; serves sigma8 as
  a derived), get_can_support_params ['Pk_grid', 'Pk_interpolator',
  'sigma8']. The w0/wa param padding block (calculate lines 355-365)
  dies in v2 — the artifact's stored input names ARE the contract.
- The legacy generator conventions: z grid = concat(linspace(0,2,100,
  endpoint=False), linspace(2,10,10,endpoint=False),
  linspace(10,50,12)) = 122; k = logspace(-4, 2, 2000) (kmax 100,
  requirement k_max 200, extrap_kmax=100 on the interpolator);
  requirements = {omegabh2, omegach2, H0, ns, As, tau, Pk_interpolator
  {z, k_max 200, nonlinear (True, False), vars_pairs delta_tot}} PLUS
  the quirk kept verbatim: "Cl": {"tt": 0} with the comment "DONT
  REMOVE THIS - SOME WEIRD BEHAVIOR IN CAMB WITHOUT WANTS_CL".
- symbolic_pofk imports (legacy lines 33-50): sys.path insert of the
  package dir, then from symbolic_pofk.linear import As_to_sigma8,
  plin_emulated; from symbolic_pofk.syrenhalofit import run_halofit,
  run_halofit_vec; wrapped in availability guards.

**Increment 1 WRITTEN, compile-clean (this window; the numpy probe
rides the next window with the staging work):**
1. DONE — emulator/geometries_grid2d.py: Grid2DGeometry (z + k axes,
   rows flattened z-outer; TARGET_LAWS_2D {none, syren_linear,
   syren_halofit} persisted by name, NO per-law state keys — the base
   is recomputed by the consumer per D-MP2-A; encode/decode =
   standardize only, torch-pure; from_targets over LAW-SPACE rows
   with the un-standardizable guard naming (z, k) points; the
   downsample persists as the STORED k grid itself, never a stride
   knob). The geometry math is the GridGeometry math at width nz*nk
   (probe_bsn1 leg 3 mirrored it green at 1D).
2. DONE — emulator/syren_base.py per D-MP2-A(3): base_pklin (k/h
   conversion, plin_emulated at z=0, the approximate-growth rescaling
   (Dz/D0)^2 (Rz/R0) at kref=1e-4 with mnu=0.06, /h^3 -> Mpc^3) and
   base_boost (As_to_sigma8 -> run_halofit_vec(return_boost=True,
   Plin_in=P*h^3)) — the legacy _compute_mps_approximation /
   _compute_boost_approximation VERBATIM (read in full this window);
   As is As_1e9 (the syren convention); import guard quiet at load,
   loud at first use. ALSO captured for the adapter port: the legacy
   low-k boost blend (k_t = 0.005 1/Mpc, n = 2:
   boost = 1 + (boost-1)*(1 - exp(-(k/k_t)^n)) — belongs in emul_mps'
   nonlinear assembly) and the legacy network-input padding block
   (w0/wa defaults — DIES in v2, the artifact's stored names are the
   contract).
## MPS status: CODE COMPLETE — awaiting the workstation board
(2026-07-11, Architect, overnight-mode continuation; probe_mps 7/7)

Steps 3-7 landed on the increment-1 foundations:

- **Generator (D-MP3):** dataset_generator_mps.py, the fourth thin
  driver — z_segments/k_log10/extrap_kmax/write_syren_base train_args
  (validated: ascending concat, nk >= 8, extrap edge >= k top;
  write_syren_base True fails AT SETUP when symbolic_pofk is absent,
  never at sample 1 of a farm); the requirements verbatim incl. the
  wants-Cl quirk comment; payload {pklin, boost[, pklin_base,
  boost_base]}; the store = 2 or 4 per-quantity 2D files + _z/_k grid
  sidecars written at alloc.
- **Staging (D-MP2-A(2)):** load_source now returns "dump_rows" (the
  sorted-unique on-disk rows of the staged set — additive key, both
  staging paths aligned); _grid2d_law_rows materializes
  log(raw / base) in RAM with the base dump row-aligned by it, applies
  k_stride (top edge always kept), REPLACES C/dv/idx row-aligned, and
  recomputes dv_mean; positivity + width guards loud. Probed through
  the REAL load_source with a working torch-stub Generator/randperm.
- **Config path:** validate_grid2d (quantity/units pairing enforced,
  the law-quantity pairing pklin<-syren_linear / boost<-syren_halofit,
  base files required-iff-syren, k_stride, transfer PERMANENT);
  from_config grid2d branch + the D-MP7 finetune sub-path (wrong-kind
  + quantity/units/law metadata at from_config, the (z, k)+stride
  equality at the build_geometry pin); build_geometry branch over the
  post-staging axes; pool_size/param_cuts/print_design/exclusivity all
  four-family aware; results.py grid2d keys (class-guarded);
  EmulatorPredictor grid2d branch (predict -> {"z", "k", quantity} as
  the reshaped LAW-SPACE surface — the consumer multiplies the base,
  D-MP2-A(4)); all four sibling adapters reject grid2d artifacts
  naming emul_mps.
- **emul_mps (D-MP4):** exactly two roots (pklin + boost), each
  self-declaring; grids must match exactly; the syren-base names ride
  the requirements when a law is in force (w/wa ride the artifact
  inputs — an absent EoS = LCDM on BOTH sides through the one
  syren_params_from rule, so generator and adapter cannot disagree);
  calculate assembles P_lin = exp(net)*base, boost = exp(net)*base
  with the base fed the EMULATED P_lin (the legacy flow) + the
  verbatim low-k blend (k_t 0.005, n 2), P_nl = B*P_lin, the legacy
  reject-on-bad-spectrum (return False) semantics; the legacy state
  keys; PowerSpectrumInterpolator ported VERBATIM (probe leg 6:
  line-for-line equal to the legacy class, Lewis attribution kept);
  get_Pk_grid / get_Pk_interpolator / sigma8 derived / 
  get_can_support_params on the captured legacy surface. The legacy
  w0/wa param-padding block is DEAD (the artifact's stored names are
  the contract).
- **Drivers (D-MP5):** sweep_ntrain_mps_emulator.py /
  tune_mps_emulator.py on family_drivers.
- **Gates (D-MP6):** mps-identity (geometry round-trips; the REAL
  staging law leg; save/rebuild/predict bitwise both laws; the
  emul_mps assembly EXACT against synthetic base stubs incl. the
  blend pinning boost->1 below k_t and the As->As_1e9 conversion
  reaching the base; pair/grid/wrong-kind guards; interpolator node
  round-trip; validate legs; D-MP7 finetune parity +
  metadata-mismatch) and mps-smoke (real CAMB: generator 200 rows
  16 z x 40 k -> pklin/boost dumps; two law-NONE trainings with the
  relative dead-network bars; the real cobaya lifecycle vs CAMB's own
  P(k, z) at 5%; the interpolator range guard). Board = 31
  (census-counted). Example YAML mps_boost_emulator.yaml.
- **Recorded decisions/interims:** (a) mps-smoke runs the law-none
  path end to end and NEVER needs symbolic_pofk — the syren assembly
  is exactly gated by the identity stubs, and the full syren + EMUL2
  hybrid run (EXAMPLE_EMUL2_EVALUATE1.yaml: cosmolike use_emulator: 2
  + emul_mps + emul_baosn + the rdrag scalar emulator) is the unit's
  ACCEPTANCE experiment, user-run on the workstation where
  symbolic_pofk lives. (b) MPS-DIAG follow-up: per-family physical
  diagnostics pages for grid2d (the D-CM9 dispatch has no _grid2d
  pages yet; the shared chi2 pages apply as-is) — small, rides POL or
  its own delta. (c) First-run risks: the Pk_grid add_requirements
  resolution onto a non-CAMB theory (the smoke's cobaya leg is
  self-diagnosing); the generator's Pk_interpolator requirement
  against the workstation cobaya version.

**The executed plan (for the record):**
3. dataset_generator_mps.py (fourth thin driver): probes ("mps",),
   EXTRA_TRAIN_KEYS z/k specs; requirements per the captured
   conventions incl. the wants-Cl quirk; payload {pklin, boost,
   pklin_base, boost_base}; store = four 2D files + grid sidecars
   (the background-driver store pattern at four files).
4. experiment.py grid2d branch (validate + staging law transform +
   build_geometry + finetune per the SPE-FT pattern; the staging
   ratio transform reads the _base files).
5. cobaya_theory/emul_mps.py (the captured legacy surface over TWO
   artifacts pklin + boost; P_nl = P_lin * B; the verbatim
   interpolator class; syren_base multiplied back per D-MP2-A(4)).
6. sweep_ntrain_mps_emulator.py / tune_mps_emulator.py
   (family_drivers).
7. Gates mps-identity / mps-smoke (+ board 29 -> 31) + example YAML +
   readme-mps-section-draft.md; EMUL2 acceptance =
   EXAMPLE_EMUL2_EVALUATE1.yaml end to end on the workstation
   (recorded as the unit's acceptance experiment, user-run).

**D-MP8 — syren VENDORED in-repo (user directive, 2026-07-11):** the
two symbolic_pofk modules the base uses (linear.py + syrenhalofit.py,
778 lines, MIT) are copied into `syren/` from the LEGACY emulmps
bundle (`emulators_code/emulmps/emulmps_emul/symbolic_pofk` — the
exact copy the legacy pipeline ran, "Vic edits" included), NOT from
PyPI. Function bodies byte-verbatim (AST-proven, 17/17); the only
deviations are import lines (two DEAD imports dropped — `warnings`
and `scipy.integrate`, neither used anywhere — and the internal
import retargeted), recorded in syren/README.md + provenance headers.
The package is now numpy-only, so: syren_base.py imports
UNCONDITIONALLY (the quiet-at-load/loud-at-use guard deleted), the
generator's write_syren_base setup check became an import sanity, and
the base ran FOR REAL on the Mac for the first time — probe: vendored
vs original bundle byte-identical over 3 cosmologies (LCDM + two
w0wa) x 5 z x 300 k (5,433 values), and base_pklin/base_boost
end-to-end through emulator.syren_base match the original-bundle
computation exactly. Gate wording updated (mps-identity/mps-smoke/
board docstrings); the stub-based assembly legs stay BY DESIGN (a
formula update must never mask an assembly bug). Re-vendoring is a
deliberate act + retrain (the change-X table row).

**Train drivers landed (2026-07-11, D-MP5 extension):**
train_cmb_emulator.py / train_baosn_emulator.py /
train_mps_emulator.py — thin wrappers over the cosmic-shear driver's
main(), which gained `main(prog, family)` + the module-level
`require_family_block(data, family, prog)` guard + the FAMILY_DRIVERS
map: each wrapper pins its data-block family, and a wrong-family YAML
fails at startup NAMING the right driver (guard probed on 8 paths:
3 pass + 5 wrong-family messages). The dispatching driver
(family=None) behaves exactly as before. README Drivers table + both
code maps updated.

**Acceptance YAML shipped (2026-07-11, POL window):** the v2 evaluate
config is `cobaya_theory/EXAMPLE_EMUL2_EVALUATE.yaml` — the legacy
lsst_y1 EXAMPLE_EMUL2_EVALUATE1.yaml with its three legacy theory
blocks (emulrdrag / emulbaosn / emulmps) replaced by emul_scalars +
emul_baosn + emul_mps; likelihood, params, and the evaluate override
kept verbatim so the legacy and v2 runs evaluate the same point. The
emulators lists carry placeholder roots under
projects/lsst_y1/emulators/ — point them at the trained rdrag /
hubble+dm / pklin+boost artifacts before running.
