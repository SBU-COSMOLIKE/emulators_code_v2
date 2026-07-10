---
name: triangle-cut-shading-all-windows
description: "Design spec (Architect, 2026-07-06): the diagnostics getdist triangle shades only the two legacy windows (omegabh2, omegam2h2) — the new omegamh2 / omegamh2ns cuts trim the sample cloud with no grey explaining the edges (user saw it on diagnostic_rescnn_t16_ntrain25000.pdf: the unshaded Omh2-marginal cliff at 0.20 and the sheared (ns, Omh2) corner at 0.17). Fix: per-window exclusion masks (not one merged mask per panel), every window drawn in the same semi-transparent grey so superposition composes the union (user's requested rendering); add the ns cut role; cover every panel where a window is 2-D-sharp incl. the derived Omh2 axis and its 1-D marginal band. Plotting-only change."
metadata:
  node_type: memory
  type: project
---

# Triangle cut shading: all four windows, same-grey superposition

Follows [[omegamh2-ns-product-cuts]] + [[param-cuts-nested-block]]. The
diagnostics triangle (`_lcdm_triangle_fig` / `_shade_cuts` /
`_cut_exclusion` in emulator/plotting.py) predates the new windows.

## Symptom (user report + Architect-verified on the PDF)

With `omegamh2 in (0.05, 0.20)` and `omegamh2ns in (0.10, 0.17)` active,
the triangle shades only the omegabh2 and Gamma^2 regions. The sample
cloud's hard edges from the new windows get no grey: the Omega_m h^2
marginal ends abruptly at 0.20, and the (n_s, Omega_m h^2) panel has a
sheared diagonal corner — both read as anomalies instead of cuts. Two code
gaps: `_cut_exclusion` implements only the legacy windows, and `ns` has no
entry in `_CUT_ROLES`.

## Design

1. **Per-window masks, not one merged mask per panel.** `_cut_exclusion`
   (or its successor) returns one boolean grid per active window that is
   sharp on that panel, instead of a single merged mask — the current
   single-mask shape cannot represent two windows excluding different
   regions of one panel.
2. **Same-grey superposition (the user's requested rendering).** Every
   window's region is filled with the same semi-transparent grey
   (one rgba, alpha chosen so a double overlap darkens visibly but a
   triple still sits clearly under the points; fills stay zorder 0 with
   the existing limit re-pinning). The union's outer edge then traces the
   true allowed region; no per-window colors, no legend growth. The
   existing single caption line stays.
3. **Coverage: every panel where a window is 2-D-sharp**, with the derived
   omegamh2 triangle column (role `omh2`) and the new `ns` role
   (`_CUT_ROLES += ("ns", ("ns",))`):

   | window     | sharp on                                              |
   |------------|-------------------------------------------------------|
   | omegabh2   | (ob, h0)                                    [existing] |
   | omegam2h2  | (om, h0); (om, omh2): g2 = om*omh2;                    |
   |            | (h0, omh2): g2 = omh2^2/(h0/100)^2          [existing] |
   | omegamh2   | (om, h0): omh2 = om*(h0/100)^2  [new];                 |
   |            | any panel with an omh2 axis: a 1-D band     [new];     |
   |            | the omh2 1-D marginal: axvspan band         [new]      |
   | omegamh2ns | (ns, omh2): product = ns*omh2               [new]      |

   The "sharp only" principle stands: no fuzzy projections of a window
   onto panels that do not determine it.
4. **One source of truth discipline.** The window formulas here mirror
   phys_cut_idx's quantity table (data_staging.py); each plotting-side
   formula carries a provenance comment pointing at that table, and the
   mask helpers stay standalone numpy functions (exec-extractable, so the
   gates run on the Mac).
5. The cuts dict remains the validated `param_cuts` sub-block the driver
   already passes (post-rename keys: omegabh2_lo/hi, omegam2h2_lo/hi,
   omegamh2_lo/hi, omegamh2ns_lo/hi).

## Validation gate

- GT-A mask correctness (Mac, numpy-only): for each window and each sharp
  panel, the per-window grid mask equals an independent reference written
  from the display formulas, on a synthetic grid; and where a panel's two
  axes fully determine a window, the panel mask evaluated at actual sample
  coordinates agrees with phys_cut_idx's per-window keep decision for
  those samples (exact cross-check against the quantity table).
- GT-B rendering smoke (Mac if getdist present, else workstation): a
  synthetic-sample triangle with all four windows active produces fills on
  exactly the panels in the coverage table (assert on the axes' artist
  lists), same rgba everywhere, plus the omh2-marginal band.
- GT-C visual (workstation, rides the pending session): regenerate the
  diagnostics PDF for the same run; every hard sample edge now has
  adjoining grey — specifically the omh2 marginal at 0.20 and the
  (ns, omh2) diagonal corner at 0.17.
- GT-D style: house scans clean; the grey-only palette keeps
  [[plots-no-red-green]] trivially satisfied; py_compile whole tree.

## Resume state (Implementer appends below)

(none yet)

### ARCHITECT_HANDOFF: READY FOR EXECUTION

- **Target file(s):** emulator/plotting.py only (`_CUT_ROLES`,
  `_cut_exclusion` -> per-window masks, `_shade_cuts` same-grey
  superposition + 1-D marginal band, docstrings).
- **Contracts & interfaces:** cuts dict keys unchanged (the validated
  param_cuts sub-block); `_shade_cuts(g, plot_names, cuts)` signature
  unchanged; internal helper shapes are yours within the per-window-mask
  goal.
- **Verbatim numerics:** the coverage-table formulas above, mirroring
  phys_cut_idx's quantity table exactly (provenance comments required).
- **Constraints & edge cases:** sharp-only shading (no fuzzy
  projections); same rgba for every window, alpha-stacked; fills under
  the points (zorder 0) with limits re-pinned; inactive windows (None
  bounds, or a missing ns column/axis) draw nothing and raise nothing;
  a window whose bounds lie outside the panel range draws nothing.
- **Validation gate:** GT-A through GT-D; paste the mask-equality output,
  the artist-assertion output, and (from the workstation) the regenerated
  triangle page in the IMPLEMENTER_HANDOFF.
- **Notes entry:** notes/triangle-cut-shading-all-windows.md (this file).
- **Next milestone:** IMPLEMENTER_HANDOFF with GT-A/B evidence; GT-C rides
  the pending workstation session (window smoke + item-27 A/B + GN-F).

### 2026-07-06 — Implementer (Opus 4.8) execution

plotting.py only, on amazing-keller. Mac dev machine has no matplotlib or
getdist, so GT-B (render) and GT-C (visual PDF) ride the workstation; GT-A is
numpy-only and was run standalone (exec-extract _window_masks from source,
plus phys_cut_idx from data_staging.py).

**Done:**

- `_CUT_ROLES` gains `("ns", ("ns",))`; new module rgba `_CUT_GREY =
  (0.55, 0.55, 0.55, 0.30)` (one grey for every window, chosen so a double
  overlap darkens but a triple stays under the points).
- `_cut_exclusion` -> `_window_masks(rx, ry, xx, yy, cuts)`: returns a LIST
  of (window_name, mask), one per active window sharp on the panel, instead
  of a single merged mask. Covers the full table: omegabh2 on (ob, h0);
  omegam2h2 on (om, h0) / (om, omh2) / (h0, omh2); omegamh2 on (om, h0) and
  as a 1-D band on any panel with an omh2 axis; omegamh2ns on (ns, omh2).
  Each formula carries a provenance comment naming phys_cut_idx's quantity-
  table helper (_omega_b_h2 / _omega_m2_h2 / _omega_m_h2 / _omega_m_h2_ns);
  exclusion is q <= lo or q >= hi, matching phys_cut_idx's strict keep.
- `_shade_cuts`: superposes one `_CUT_GREY` contourf per sharp window
  (zorder 0 under the points, separate artists alpha-stack, limits
  re-pinned) instead of one light fill; a new `_shade_omh2_marginal` bands
  the omegamh2 window on the derived omh2 diagonal marginal with axvspan
  (drawing nothing when the window is off or a bound sits outside the panel
  range). The single caption line is unchanged.
- `_lcdm_triangle_fig` cuts-docstring updated to the full param_cuts key set.
  Inactive windows / a missing ns axis / out-of-range bounds all draw
  nothing and raise nothing (None-guarded, all(...) drops empty masks).

**Gate evidence (raw):**

- GT-A.1 mask == independent reference (synthetic grid, every sharp panel,
  both axis orders): omegabh2 3330 cells (both orders), omegam2h2 (om,h0)
  2660 / (om,omh2) 1923 / (h0,omh2) 2180, omegamh2 (om,h0) 2270, omegamh2
  band (ns,omh2) 1740, omegamh2ns 2691 — all array-equal to the display-
  formula reference; a non-sharp (ob,om) panel returns 0 masks. PASS.
- GT-A.2 cross-check at 20000 sample coordinates vs phys_cut_idx's per-window
  keep (single window isolated with omegabh2_hi = 1e9): each window's mask
  == ~phys keep, exactly — omegabh2 1010==1010, omegam2h2 8394==8394,
  omegamh2 6642==6642, omegamh2 band 6642==6642 (confirming the derived omh2
  column equals phys's _omega_m_h2), omegamh2ns 12430==12430. PASS.
- GT-B rendering smoke: NOT run (no matplotlib/getdist on the Mac). Rides the
  workstation: a synthetic-sample triangle with all four windows active
  should show contourf artists on exactly the coverage-table panels, same
  rgba, plus the omh2-marginal axvspans.
- GT-C visual: workstation-deferred (regenerate the diagnostics PDF; grey
  should adjoin the omh2-marginal cliff at 0.20 and the (ns, omh2) 0.17
  corner).
- GT-D style: house scans clean on the touched tree (caps 0, ` -- ` 0,
  legend-less 0, non-hot comprehensions 0, lines>90 0); grey-only palette
  keeps plots-no-red-green trivially satisfied; whole-tree py_compile clean.
  PASS.

Open: GT-B + GT-C, both on the pending workstation session.

### 2026-07-06 — Architect re-audit: ACCEPTED (GT-B/GT-C deferred)

Verified independently (own harness, own reference formulas, different
seed): _window_masks matches the display-formula references on every sharp
panel in both axis orders; the (ns, omh2) product region and the omh2 band
are exact; a non-sharp panel and an all-None cuts dict yield zero masks
(inactive = silent); and each window's panel mask complement equals
phys_cut_idx's keep decision at 20k sample coordinates, exactly. The
_shade_omh2_marginal bounds-outside-range guards are correct (no inverted
axvspan). Scans clean. Deviations: none claimed, none found; the
_shade_omh2_marginal helper sits within the granted internal-shape latitude.

Open: GT-B (artist-level render smoke) + GT-C (regenerated diagnostics PDF),
both workstation. The consolidated deferred queue is now: G1 runtime import,
item-27 ci.init_probes A/B, G-F window smoke, GN-F nested-block load, GT-B,
GT-C — one train_single run with the nested YAML + a tight omegamh2 window +
--diagnostic covers G-F + GN-F + GT-C + the import leg in a single shot; the
item-27 A/B and the GT-B artist assertions are two extra small scripts.

### 2026-07-08 — board status (Architect): triangle-shading NOT RUN (optional)
The board's one optional gate; every required gate is green (runs 10
and 11). The acceptance here is the Architect eyeballing the
gates_diag_*.pdf shading from the production-diagnostic run — relay
pending. No code question is open against this note.

## GT-C visual verdict (Architect eyeball, 2026-07-10)

Relayed PDF: gates_diag_resmlp_t16_ntrain25000.pdf (the run-10-era
production-diagnostic gate output; config = the shipped example YAML).
Rendered per page and zoomed on the acceptance panels.

**PASS for every window the run activated; the two NAMED features do not
exist in this config, and correctly so.** The shipped example YAML carries
omegamh2_*/omegamh2ns_* COMMENTED OUT — only omegabh2 (0.014, 0.035) and
omegam2h2 (0.015, 0.08) were active. Observed:

- (H0, Omega_b): grey above and below the allowed band, adjoining the
  cloud — the omegabh2 window, 2-D-sharp exactly where the coverage table
  says. PASS.
- (H0, Omega_m), (H0, omh2), (Omega_m, omh2): grey wedges adjoining the
  cloud edges — the omegam2h2 window. PASS.
- Panels where no active window is 2-D-sharp (all As / ns columns):
  correctly unshaded. PASS.
- omh2 1-D marginal: NO band, and the KDE runs smoothly past 0.20 to
  ~0.25 — CONSISTENT: omegamh2 was not active (and omegam2h2 = (Omega_m
  h)^2 is not a pure function of Omega_m h^2, so no 1-D band is expected).
  The absent 0.20 cliff / (ns, omh2) 0.17 shear are therefore not
  failures; those features require the two commented windows.

**Closing the named 0.20/0.17 items without a retrain:** the registered
`triangle-shading` gate (gates/checks/gt_b_triangle.py, off the default
sweep, "not run" on the board) is the machine version — synthetic samples,
ALL FOUR windows active including omegamh2ns (0.10, 0.17), asserting grey
fills on exactly the coverage-table panels plus the omh2-marginal axvspan
bands. One run of `python gates/run_board.py --gate triangle-shading`
closes GT-B and, with this eyeball, GT-C. (The alternative — uncommenting
the two windows and regenerating the 25k-row diagnostic PDF — would
re-prove the same coverage visually at training cost; not required.)

## GT-B first execution (2026-07-10) + delta D-GTB-1 (Architect, applied directly)

First run ever of the triangle-shading gate (it was registered but off
the default sweep, and the Mac has no matplotlib/getdist, so the check
script shipped without ever executing). Result: 2/3 assertions PASS —
grey fills on 7 panels, the omh2 axvspan bands present — and one FAIL
that is the CHECK'S OWN classifier bug, not a plotting bug: the check
assumed "points draw as lines, not filled patches" and counted 11
off-grey filled artists, but _lcdm_triangle_fig deliberately draws every
off-diagonal panel as a viridis-coloured SCATTER (a filled
PathCollection) plus a shared colorbar. The 11 off-grey artists are the
data itself, which is supposed to be coloured. The plotting fix under
test is correct (also confirmed by the same-day Architect eyeball of the
real diagnostics PDF, above).

**Delta D-GTB-1 (compile-checked + stub-probed):** the classifier now
selects the shading layer by the design contract instead of guessing
artist types — plotting._shade_cuts draws every cut fill (contourf
regions and axvspan bands) at zorder 0, deliberately under the data, so
facecolors() filters artists to zorder == 0 and the span count does the
same. The stub probe confirms: scatter/colorbar/legend artists ignored;
zorder-0 grey counted; a genuinely mis-coloured zorder-0 fill still
fails the gate (the assertion keeps its teeth).

**Honest margins, recorded:** (1) the fix ships Mac-blind (no
matplotlib here) — the stub probe covers the classifier logic, the
workstation rerun is the proof; (2) the shipped GT-B asserts shading
PRESENCE (>0 panels + bands), not the spec's "exactly the
coverage-table panels" — the exact panel set was verified visually by
the Architect eyeball for the windows the real config activates; a
future tightening could assert the exact set. Rerun:
`python gates/run_board.py --gate triangle-shading`.
