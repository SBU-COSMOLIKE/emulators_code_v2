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
**Next steps (fresh context resumes here):**
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
