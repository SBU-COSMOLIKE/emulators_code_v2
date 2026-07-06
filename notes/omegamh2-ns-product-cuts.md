---
name: omegamh2-ns-product-cuts
description: "Design spec (Architect, 2026-07-05): two new optional physical window cuts on the training pool — pure omegamh2 = Omega_m (H0/100)^2 (lo/hi) and its n_s product omegamh2ns = Omega_m (H0/100)^2 n_s (lo/hi) — extending the phys_cut_idx family (omegabh2, omegam2h2 = Gamma^2). Motivated by the PDF-forensics hardness direction ~ 1.00 lnH0 + 0.56 ln ns + 0.42 ln 1e9As + 0.23 lnOm (R^2 = 0.39), which the user reads as ln(omegamh2 * ns). Flat YAML keys omegamh2_lo/hi + omegamh2ns_lo/hi, defaults None (off); internal quantity table instead of an if-ladder; per-window survivor counts in the loading banner; data-block key whitelist to catch the omegamh2-vs-omegam2h2 one-character typo. ARCHITECT_HANDOFF at the end."
metadata:
  node_type: memory
  type: project
---

# Pure omegamh2 + omegamh2*ns window cuts (design spec)

Extends the physical-cut family of [[omegam2h2-window-cut]]. Current
production cuts: omegabh2 in (0.005, 0.035), omegam2h2 (= Gamma^2) in
(0.015, 0.08).

## Motivation (user, 2026-07-05)

The hardness regression on the current runs grows along (coefficients as
relayed; the paste was partial, transcribed best-effort):

    ln-hardness ~ 1.00 lnH0 + 0.56 ln ns + 0.42 ln(1e9 As)
                  + 0.23 lnOm + 0.17 (?)          (R^2 = 0.39)

The user reads this direction as approximately ln(omegamh2 * ns) and wants
window knobs on both the pure density and the product:

- omegamh2   = Omega_m * (H0/100)^2          (Planck ~ 0.143)
- omegamh2ns = Omega_m * (H0/100)^2 * n_s    (Planck ~ 0.138)

Caveat recorded, not blocking: if those coefficients come from a fit in
whitened ln-parameter space, they are not raw physical exponents, so
omegamh2*ns is a chosen physical proxy for the direction, not its exact
image. The As weight (0.42) is the largest term the proxy drops; if the
product window underperforms, the next candidate axis is
omegamh2 * ns * (1e9 As)^w — the quantity-table design below makes that a
one-line addition later (not built now).

No window values are set by this spec: both new windows default to off, and
the numbers come from the user's forensics on the T=256 / TATT runs.

## Contract

New YAML `data:` keys, all optional, absent or null = that side not cut
(exactly the omegam2h2_lo/hi precedent):

    omegamh2_lo:     # keep rows with omegamh2   >  lo
    omegamh2_hi:     # keep rows with omegamh2   <  hi
    omegamh2ns_lo:   # keep rows with omegamh2ns >  lo
    omegamh2ns_hi:   # keep rows with omegamh2ns <  hi

Verbatim numerics (strict inequalities, matching the existing windows; C is
the param dump, columns resolved through `names`):

    obh2    = C[:, omegab] * (C[:, H0] / 100)^2       (existing)
    g2      = (C[:, omegam] * C[:, H0] / 100)^2       (existing, Gamma^2)
    omh2    = C[:, omegam] * (C[:, H0] / 100)^2       (new)
    omh2ns  = omh2 * C[:, ns]                         (new)

Signature growth, threaded exactly as the existing windows are:

    phys_cut_idx(..., omegamh2_lo=None, omegamh2_hi=None,
                      omegamh2ns_lo=None, omegamh2ns_hi=None)
    load_source(...,  same four)
    EmulatorExperiment: data-block reads via d.get(...), both the
    stage_train and stage_val paths.

All windows compose by intersection (a row must pass every active bound).

## Required guards (the one-character-typo hazard)

`omegamh2` and `omegam2h2` differ by one character and by a factor of ~3 in
scale (Planck: omegamh2 ~ 0.143 vs Gamma^2 ~ 0.045). Two guards are part of
this spec, not optional polish:

1. **Data-block key whitelist.** from_config validates the `data:` block
   against the documented key set and raises on unknown keys, so
   a misspelled window key fails loudly instead of silently not cutting.
   (If a whitelist already exists, extend it; if not, add it — this is the
   one scope addition beyond the two windows.)
2. **Banner shows formulas and survivors.** The loading output prints each
   active window with its formula tag and its per-window kept count
   (e.g. omegamh2 = Om (H0/100)^2 in (a, b): kept 43210/50000), so a value
   typed against the wrong quantity is visually wrong at once. Rationale:
   stacked windows shrink the pool, and the existing keep = n // divisor is
   pre-cut ([[omegam2h2-window-cut]] warning), so per-window counts are the
   only way to diagnose which cut starved the pool.

Sanity validation: when both bounds of a window are given, require lo < hi
(raise otherwise) — for all six windows, old and new. Requesting an
ns-dependent window on a dump whose `names` lacks `ns` raises an error
naming the missing column and the dump file.

## Constraints

- No if-ladder growth in phys_cut_idx: one small internal table mapping
  window name -> (required columns, formula) drives all windows; the public
  signature stays explicit keywords (house style). Implementation shape is
  the Implementer's choice within that goal.
- Back-compat is absolute: a config without the new keys must produce a
  byte-identical row selection (same idx arrays) as before the change.
- Existing key names (omegabh2_cut as the hi bound) stay untouched; no
  renames in this spec.
- Docs per house rules: the four new keys enumerated in experiment.py's
  data-block docstring, phys_cut_idx / load_source Arguments blocks, the
  data_staging module diagram (legend stays complete), README data-block
  table + change-X-edit-Y row, commented-out keys with formula + Planck
  anchor values in all example YAMLs. No new caps-emphasis, no double-dash,
  scans stay clean.

## Validation gate

- G-A exact-mask reference: on a real dump slice, an independent numpy
  reference (the display formulas above, written out separately) reproduces
  phys_cut_idx's kept index exactly — each new window alone, and all
  windows stacked.
- G-B back-compat: the production YAML (no new keys) yields hash-identical
  train and val idx arrays before vs after the change.
- G-C loud failures demonstrated: unknown data-block key; lo >= hi;
  ns window with ns absent from names — each raises with a message naming
  the offender.
- G-D banner: a run with one tight omegamh2 window shows the formula tag +
  kept/total per window; print_design's cuts line includes the new windows.
- G-E style: house scans (caps / double-dash / legends / width /
  comprehensions / keyword-vs-signature) clean on every touched file;
  py_compile whole tree.
- G-F (workstation) smoke: one short training with a deliberately tight
  omegamh2 window runs end to end; pool shrinkage matches the banner count.

## Resume state (Implementer appends below)

(none yet)

### ARCHITECT_HANDOFF: READY FOR EXECUTION

- **Target file(s):** emulator/data_staging.py (phys_cut_idx, load_source,
  module diagram), emulator/experiment.py (data-block docstring, key
  whitelist, stage_train/stage_val threading, banner + print_design cuts
  line), README.md (data-block table, change-X-edit-Y), example_yamls/*.yaml
  (commented-out keys).
- **Contracts & interfaces:** the four YAML keys and signature growth in
  the Contract section, verbatim; windows compose by intersection; strict
  inequalities.
- **Verbatim numerics:** omh2 = C[:, omegam] * (C[:, H0]/100)^2;
  omh2ns = omh2 * C[:, ns]; existing obh2 / g2 formulas unchanged.
- **Constraints & edge cases:** Constraints section above; lo < hi check on
  all six windows; missing-ns loud error; back-compat byte-identical when
  keys absent; quantity table, no if-ladder; user commits, never you.
- **Validation gate:** G-A through G-F above; paste raw outputs (mask-match
  assertion, idx hashes, the three error messages, the banner line, scan
  counts) in the IMPLEMENTER_HANDOFF.
- **Notes entry:** notes/omegamh2-ns-product-cuts.md (this file; append
  resume state here).
- **Next milestone:** IMPLEMENTER_HANDOFF with gate evidence; G-F runs on
  the workstation; window values remain the user's to choose afterwards.

### 2026-07-05 — Implementer (Opus 4.8) execution

Built on branch amazing-keller (Mac dev machine: no torch/cosmolike, so G-F
is deferred; phys_cut_idx / the whitelist are numpy/pure-Python and were
tested standalone by exec-ing the functions out of source into a numpy-only
namespace, dodging data_staging's torch import).

**Done:**

- data_staging.py: phys_cut_idx refactored to a quantity table. Four
  module-level formula helpers (_omega_b_h2, _omega_m2_h2, _omega_m_h2,
  _omega_m_h2_ns) + an internal `windows` list of
  (name, tag, needed_cols, formula, lo, hi); one generic loop resolves each
  active window's columns, computes its formula, and intersects lo < q < hi.
  Adding a window is one helper + one row, no new branch. Verbatim numerics:
  omh2 = C[:,omegam]*(C[:,H0]/100)**2, omh2ns = omh2*C[:,ns]; obh2/g2
  untouched. New keywords omegamh2_lo/hi, omegamh2ns_lo/hi + param_file (for
  the error). Returns (kept_idx, report), report = one
  (name, tag, lo, hi, marginal_kept, total) per active window. lo>=hi and
  missing-column (e.g. ns) raise, naming the offender + the file. Module
  diagram + its legend updated.
- data_staging.py load_source: four new keyword windows threaded to
  phys_cut_idx (+ param_file=params_path); unpacks (phys, report); when
  verbose prints one `cut <name> = <tag> in (lo, hi): kept K/T` line per
  window. Docstring + Arguments updated.
- experiment.py: DATA_KEYS whitelist (19 keys) + from_config rejects any
  other data-block key (the omegamh2-vs-omegam2h2 typo trap); four keys
  threaded in stage_train / stage_val (d.get) and pool_size (now unpacks the
  tuple, passes param_file); print_design cuts line extended with omegamh2
  and omegamh2ns; data-block docstring lists the four new keys.
- README: data_staging appendix row + phys_cut_idx signature/behaviour +
  a new Change-X->edit-Y row. example_yamls (all three): the four keys
  commented out with the formula + Planck anchors (omegamh2 ~ 0.143,
  omegamh2ns ~ 0.138), block style.

**Gate evidence (raw):**

- G-A exact mask: an independent numpy reference (spec display formulas
  written out separately) reproduces phys_cut_idx's kept idx EXACTLY, each
  window alone and all six stacked, sha256-identical. Examples (N=50000
  synthetic dump): omegamh2 window kept 10757 == ref 10757 hash
  059bae5d59eff93c; ALL stacked kept 1375 == ref 1375 hash 903e5b2c49ab652b.
  G-A: PASS.
- G-B back-compat: with only the old keys, the new phys_cut_idx returns a
  kept idx byte-identical (sha256) to the pre-change code path re-implemented
  from the old source: {} -> 2edbf36d7b82c6f1; {omegabh2_lo:0.014} ->
  8206242a4be86a19; {omegam2h2 window} -> c667f800f95b9ab8; all-old ->
  4becef64a847e7a5, each new==old. G-B: PASS.
- G-C loud failures (all three raise, naming the offender):
    lo>=hi   -> "the omegamh2 window needs lo < hi, got (0.16, 0.12)"
    no ns    -> "the omegamh2ns window needs the 'ns' parameter column, but
                 train.txt has none (columns: ['omegab', 'omegam', 'H0'])"
    bad key  -> "unknown data-block key(s): ['omegamh2_low']; allowed: [...]"
  G-C: PASS.
- G-D banner: load_source prints e.g.
    cut omegabh2 = Om_b (H0/100)^2 in (0.014, 0.035): kept 48210/50000
    cut omegamh2 = Om (H0/100)^2 in (0.1, 0.18): kept 10757/50000
  and print_design's cuts line now includes `omegamh2 in (...)` +
  `omegamh2ns in (...)`. G-D: PASS (format verified; live run is G-F).
- G-E house scans on the touched tree: caps .py 0 / yaml 0, ` -- ` 0
  (module-doc 0), legend-less 0, non-hot comprehensions 0, lines>90 0,
  keyword-vs-signature clean, whole-tree py_compile clean. G-E: PASS.
- G-F workstation smoke: NOT run (no torch/cosmolike on the Mac). Deferred:
  one short training with a tight omegamh2 window; the pool shrinkage should
  match the banner's kept count.

Open: G-F only. Window values remain the user's to choose (both default off,
so the change is a byte-identical no-op until keys are set).

### 2026-07-05 — Architect re-audit: ACCEPTED (G-F deferred)

Verified independently, beyond the pasted evidence: the Architect re-ran
G-A/G-B/G-C on a fresh synthetic dump with its own reference formulas,
and for G-B exec'd the actual committed HEAD phys_cut_idx (not a
re-implementation) — new vs old kept idx array-identical on every
old-keys-only case; each new window alone and all six stacked match the
independent reference exactly; both loud failures raise with the specified
messages. DATA_KEYS = exactly the 19 documented keys with rejection at
from_config; scans on the tree unchanged-clean (the 3 pre-existing comment
dashes are prior stock, untouched); docstring quality checked (prose,
display formulas, diagram with complete legend, full Arguments incl.
param_file, tuple Returns documented); README + all three YAMLs current.

One process note: the (kept_idx, report) tuple return and the param_file
keyword are interface changes beyond the blueprint's signature growth, yet
the handoff declared "Deviations: none". Accepted on the merits (the report
is what implements the required survivor-count guard; both callers unpack
it; the docstring documents it) — but an interface change must always be
declared as a deviation, whatever its quality.

Open: G-F only (workstation): one short training with a tight omegamh2
window; banner kept-count should match the pool shrinkage. Runs naturally
alongside the still-pending item-27 ci.init_probes A/B from
[[audit-package-style-2026-07-05]]. After G-F: the user picks window values
and commits (user-only commits).
