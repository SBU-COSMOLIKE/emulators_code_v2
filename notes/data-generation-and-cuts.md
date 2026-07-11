# Data generation, staging, and the physical parameter cuts

Consolidated 2026-07-11 from compute-data-vectors-import.md,
param-cuts-nested-block.md, omegam2h2-window-cut.md,
omegamh2-ns-product-cuts.md, triangle-cut-shading-all-windows.md,
data-staging-ram-and-source-dict.md (retired; full texts in git
history). Code homes: compute_data_vectors/, emulator/data_staging.py.
The user-facing story is README section 18 ("Generating the training
set" — the four-generator table + the shared-core walkthrough).

## The generators (compute_data_vectors/)

- One shared core, `generator_core.py` (D-CM3-A): the CLI (15 flags),
  the tempered/uniform sampling, the MPI master/worker farm, the
  checkpointing, capture_native_output. Four thin drivers subclass it:
  dataset_generator_lensing.py (cosmolike dv), _cmb.py (four spectra
  files _tt/_te/_ee/_pp), _background.py (the _h/_dm pair + _z.npy
  sidecars; desert-overlap loud), _mps.py (pklin/boost surfaces + the
  syren _base files when write_syren_base + _z/_k sidecars; the
  wants-Cl CAMB quirk kept verbatim). Fifth tool:
  compute_cmb_covariance.py (the analytic per-multipole covariance;
  families-scalar-cmb.md has its physics).
- The lensing generator was imported VERBATIM from the user's
  production copy (byte-identical except one header path) —
  battle-tested production code gets NO house-style retrofit; the
  README, not the script, carries the didactic layer.
- Two sampling modes, peers: tempered (--unif 0) samples
  log p_T = [-(1/2)(theta-theta_0)^T Sigma~^-1 (theta-theta_0)
  + log pi(theta)] / T with Sigma~ = the params covmat, correlations
  CLIPPED to |corr| <= maxcorr (default 0.15); uniform (--unif 1)
  fills the temperature-stretched box (bounds widened T*width/5; lnp
  = 1; files tag `_<probe>_unifs`). WHY each knob: T widens the cloud
  past the posterior (chains at the edge must stay inside the training
  support); maxcorr fattens the Fisher pancake PERPENDICULAR to
  degeneracies; --boundary < 1 shrinks val/test INSIDE the training
  support (accuracy dies at the cloud edge).
- The output contract (closes every loop):
  `<paramfile>_<probe>_<T>.1.txt` (weights/lnp/params/chi2*; staging
  drops the bookends), `.paramnames` (first column = cobaya names ->
  ParamGeometry -> h5 -> get_requirements — THE naming loop),
  `.covmat` (the INPUT whitening basis), `.ranges`, the dv store
  (per-family file set), `<failfile>` (recompute with --loadchk 1).
- Loud disambiguation: the generator YAML is a COBAYA yaml with its
  OWN train_args (probe/ord/fiducial/params_covmat_file) — unrelated
  to the trainer's train_args. Two stages, two schemas.

## Staging (emulator/data_staging.py)

- Memmap the dv dump (never loaded whole); params load to RAM;
  phys_cut_idx is params-only. `stage_source(C, dv, idx, ram_frac)`:
  if the sorted-unique row subset fits ram_frac of free RAM,
  materialize the COMPACT copy and reindex locally; else keep the
  memmap with global idx — invisible to consumers because everything
  touches C/dv THROUGH idx. The source dict carries the centers
  (C_mean, dv_mean via chunked stream_stats).
- Multi-GPU rule: parallel paths set ram_frac = 0 — np.asarray on a
  shared memmap makes a PRIVATE per-worker copy (the shared-budget
  trap in parallel form).
- `load_source` returns `dump_rows` (sorted-unique disk rows) so a
  sibling dump file — the MPS syren base dumps — stays row-aligned
  through a shuffled staging.
- The .paramnames cross-check: when the sidecar exists, its first
  column must match the covmat-header names ORDER INCLUDED; mismatch
  is a loud error naming both lists.

## The physical cuts (data.param_cuts)

- Schema: nested `data.param_cuts:` block, whitelist of 8 keys;
  `omegabh2_hi` REQUIRED; omegabh2_lo, omegam2h2_lo/hi, omegamh2_lo/
  hi, omegamh2ns_lo/hi optional (absent = that side open). Legacy flat
  keys / `omegabh2_cut` raise the paste-ready migration ValueError.
- The formulas (verbatim, strict inequalities, intersection-composed):
  obh2 = omegab*(H0/100)^2; g2 = (omegam*H0/100)^2 (= Gamma^2, the
  transfer shape parameter; Planck ~0.045); omh2 = omegam*(H0/100)^2
  (Planck ~0.143); omh2ns = omh2*ns (Planck ~0.138). Implementation is
  a QUANTITY TABLE (name, tag, needed cols, formula, lo, hi) — adding
  a window = one helper + one row, never an if-ladder. One-character
  traps (omegamh2 vs omegam2h2, ~3x scale apart) are why the whitelist
  is loud.
- THE FRAMING LESSON (the user's coverage argument beat the
  failure-targeted fit): n_train draws from the CUT pool, so cutting
  rarefied volume DENSIFIES the kept region instead of costing data —
  a coverage cut, not a failure cut. Evidence: sparsity and failure
  rate are U-shaped along omegam^2h^2 (the sampling cloud's long axis
  in (lnOm, lnh)); the (0.015, 0.08) window removed 96% of dchi2>100
  and took frac>0.2 from 0.59 to 0.156 same-day.
- Post-cut signature: residual failures HUG the cut boundaries
  (one-sided training support at the window edge) — the motivation for
  the generator's --boundary < 1.
- Live production values (2026-07-03, user-confirmed): omegabh2
  (0.005, 0.035), omegam2h2 (0.015, 0.08); omegamh2/omegamh2ns windows
  exist but ship commented out in the example YAML.
- The banner prints each active window's formula tag + per-window kept
  count (the only starvation diagnostic under stacked windows); the
  pool guard raises "physical pool too small after param_cuts" naming
  kept/requested and the remedy.

## Cut shading on the diagnostics triangle

Every active window shades sharp panels in ONE shared semi-transparent
grey `_CUT_GREY = (0.55, 0.55, 0.55, 0.30)` at zorder 0 (superposition
composes the union); sharp-only principle — no fuzzy projections onto
panels that do not determine a window; each plotting-side formula
carries a provenance comment naming phys_cut_idx's quantity-table
helper (ONE source of truth). D-GTB-1 lesson: the gate classifies the
shading layer by its design contract (zorder 0), never by rendering
heuristics.
