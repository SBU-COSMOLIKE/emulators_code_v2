# The background and matter-power output families (BSN + MPS)

Consolidated 2026-07-11 from baosn-emulators.md and mps-emulators.md
(retired; full delta ledgers in git history). User-facing stories are
README sections 16 and 17; the EMUL2 acceptance config is
cobaya_theory/EXAMPLE_EMUL2_EVALUATE.yaml.

## BSN — the expansion history. CODE COMPLETE; gates bsn-identity/bsn-smoke.

- Headline: only H(z) is a network; every distance is IMPOSED physics
  computed from it. The two-regime ruling (D-BSN3-A, the user's call:
  the recombination distance is the discontinuity): (1) an H(z)
  artifact on the SN window z in [0, ~3] — distances inside by
  cumulative Simpson of c/H; (2) a SECOND artifact of D_M trained
  DIRECTLY on the recombination window z in ~[1000, 1200]; (3) the
  legacy analytic z->1200 extension is DEAD (it existed only for want
  of a recombination emulator); (4) getters serve PIECEWISE by query
  redshift and a desert query is a LOUD error naming both windows —
  never a silent bridge.
- GridGeometry (geometries/grid.py): ScalarGeometry math at width NZ
  + a persisted z grid, quantity, units, and a TARGET LAW by name —
  TARGET_LAWS {none, log_offset(offset)}, the law INSIDE
  encode/decode (encode = standardize(log(y + offset))); from_targets
  applies the law FIRST. H trains under log_offset; D_M raw.
- emulator/background.py (single-sourced for the adapter, the
  diagnostics, and scripting): cumulative_simpson VERBATIM from the
  legacy (the recorded FINDING: exact at the even doubled-grid points
  — every served point — but a half-chunk O(dz^3) approximation at
  odd points, ~6.5e-4 on the probe cubic; kept bug-for-bug, only
  shades interpolation); doubled grid linspace(0, z_max, 2NZ+1); flat
  conversions dl = chi(1+z), da = dl/(1+z)^2; C_KMS = 2.99792458e5.
  FLAT ONLY V1: omk is loud. The legacy CURVATURE branch was
  dimensionally WRONG (sinh(chi*K_abs)/K_abs with K_abs =
  |omk|(H0/c)^2; correct is sinh(sqrt(K) chi)/sqrt(K)) and is
  deliberately not reproduced — a corrected branch is a future spec
  with a CAMB-comparison gate leg at omk != 0.
- Generation: dataset_generator_background.py — one background-only
  CAMB pass yields BOTH quantities; z_sn/z_rec specs with the
  no-overlap desert rule; grids written ONCE as _z.npy sidecars
  (training consumes the grid from a FILE).
- emul_baosn: exactly two self-declaring roots (Hubble km/s/Mpc +
  D_M Mpc); window layout persisted, desert-loud at startup; getters
  get_Hubble (km/s/Mpc or 1/Mpc), get_comoving_radial_distance,
  get_angular_diameter_distance (+ the two-redshift variant
  (chi2-chi1)/(1+z2)), get_luminosity_distance. rdrag comes from the
  SCALAR family — a sampling YAML pairs the theories.
- Diagnostics: per-window residual bands with the desert marked + the
  derived-distance page computed through the REAL pipeline
  (pipeline(pred H) vs pipeline(true H)).
- bsn-smoke checks the served values against CAMB'S OWN background at
  an off-center point (2%) — truth available, the strongest
  end-to-end test on the board.
- Never re-propose: the analytic extension; the legacy curvature
  formula; PCA/TMAT compression in V1; transfer over BSN (PERMANENT);
  porting legacy .pt artifacts (retrain); a silent desert bridge.

## MPS — the matter power spectrum. CODE COMPLETE; gates mps-identity/mps-smoke.

- Headline (D-MP2): the network learns the CORRECTION to an analytic
  formula — target = log(P / P_base); two artifacts: pklin (base =
  the syren linear formula, Mpc^3) and boost (base = syren-halofit,
  B = P_nl/P_lin); P_nl = B * P_lin at serving. Law registry
  TARGET_LAWS_2D {none, syren_linear, syren_halofit} by name;
  law-quantity pairing enforced.
- D-MP2-A (where the cosmology-dependent base lives — the ruling that
  shaped the unit): never inside geometry encode/decode (a per-batch
  GPU sync, or a torch-port drift channel against the very formula
  being corrected). Instead: (1) the GENERATOR writes the base beside
  the raw dump (*_base.npy files) — the consumed base is ON DISK;
  (2) STAGING forms log(raw/base) once, cold (_grid2d_law_rows,
  row-aligned through load_source's dump_rows key; k_stride thinning,
  top edge always kept); (3) ONE base module emulator/syren_base.py
  (base_pklin: k/h conversion, plin_emulated at z=0, approximate
  growth (Dz/D0)^2 (Rz/R0) at kref=1e-4, mnu=0.06, /h^3; base_boost:
  As_to_sigma8 -> run_halofit_vec(return_boost=True, Plin_in=P*h^3));
  syren_params_from is the ONE parameter-mapping rule (As or As_1e9;
  an absent equation of state means LCDM — a model fact) shared by
  generator and adapter; (4) the predictor's grid2d branch returns
  the LAW-SPACE surface — the base multiply-back is emul_mps's job.
- D-MP8 — syren VENDORED: `syren/` holds the two symbolic_pofk
  modules copied from the LEGACY emulmps bundle (the user's own
  edits included — never PyPI); function bodies AST-byte-verbatim;
  only import-line deviations (two dead imports dropped, one
  retargeted), recorded in syren/README.md; numpy-only, so
  syren_base imports UNCONDITIONALLY and the base math runs on the
  Mac. Vendored == original proven byte-identical over 5,433 values.
  Re-vendoring is a deliberate act + RETRAIN.
- Grid2DGeometry (geometries/grid2d.py): two stored axes (z, k), rows
  flattened z-outer; standardize-only in law space; the k downsample
  persists as the stored grid itself. Legacy grids recorded: z = 122
  points (three linspace segments 0-2-10-50), k = logspace(-4, 2,
  2000), extrap_kmax 200.
- Generation: dataset_generator_mps.py — the verbatim CAMB quirk
  `"Cl": {"tt": 0}` ("DONT REMOVE THIS - SOME WEIRD BEHAVIOR IN CAMB
  WITHOUT WANTS_CL") kept; write_syren_base fails AT SETUP when the
  base cannot be formed.
- emul_mps (the EMUL2 provider): PowerSpectrumInterpolator ported
  VERBATIM from the legacy (adapted from CAMB by Antony Lewis,
  attribution kept; proven line-for-line); P_lin = exp(net)*base; the
  boost base is fed the EMULATED P_lin (the legacy flow); the
  verbatim low-k blend k_t = 0.005 1/Mpc, n = 2: boost = 1 +
  (boost-1)(1 - exp(-(k/k_t)^n)); get_Pk_grid / get_Pk_interpolator /
  sigma8. The legacy w0/wa param-padding block is DEAD — the
  artifact's stored input names ARE the contract.
- EMUL2 hybrid inference: cosmolike `use_emulator: 2` consumes
  emulated CAMB products — emul_mps + emul_baosn + the scalar rdrag,
  three theories in one YAML. The unit's ACCEPTANCE experiment is the
  full EMUL2 evaluate run, user-run on the workstation; the shipped
  config is cobaya_theory/EXAMPLE_EMUL2_EVALUATE.yaml (the legacy
  yaml with the three legacy theory blocks replaced; likelihood/
  params/override kept verbatim so both runs evaluate the same
  point; placeholder artifact roots to fill).
- MPS-DIAG (closed 2026-07-11): grid2d diagnostics pages. THE key
  fact: in law space the residual pred - truth = ln(P_pred/P_truth)
  — THE BASE CANCELS — so the pages read directly as fractional
  error of the served spectrum (law none = plain fractional;
  res_kind names which). grid2d_residual_diagnostic returns the
  (nz, nk) median-|res| surface, per-k bands at first/middle/last z
  (deduplicated for small nz), and the worst-chi2 surface;
  _grid2d_pages = median + worst heatmaps on a SHARED viridis scale
  + stacked per-k band panels with the worst overlay; mps-smoke's
  boost training carries the leg.
- mps-smoke runs the law-NONE path end to end vs CAMB's own P(k,z)
  (5%); the syren assembly is gated by mps-identity's closed-form
  STUB bases — deliberate, so a formula update can never mask an
  assembly bug.
- First-run risks: Pk_grid requirement resolution onto a non-CAMB
  theory; the generator's Pk_interpolator requirement vs the
  workstation cobaya version.
- Never re-propose: the base inside encode/decode; a torch port of
  the syren formulas; recomputing the base at staging or serving; the
  w0/wa padding block; porting .keras weights (retrain); transfer
  over MPS (PERMANENT); removing the wants-Cl quirk.
