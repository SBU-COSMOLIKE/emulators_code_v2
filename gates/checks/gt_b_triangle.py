#!/usr/bin/env python3
"""triangle-shading (optional): the corner plot is honest.

WHAT: the grey cut-window shading on the diagnostics corner plot. WHY:
the shading tells a reader which parameter regions the density cuts
removed; a wrongly placed or wrongly coloured fill misleads exactly
where the plot is supposed to warn. HOW: render a synthetic-sample
triangle headlessly (matplotlib Agg) through the real
plotting._lcdm_triangle_fig with all four cut windows active, then
check the drawn artists on each panel: grey fills appear on the 2-D
panels, every fill uses the one shared colour (plotting._CUT_GREY),
and the omh2 1-D marginal carries a vertical band. Prints the
per-panel counts, exits nonzero on any failure. Needs only getdist +
matplotlib (spec: data-generation-and-cuts.md:72-75).

PS: the omh2 marginal = the 1-D diagonal panel of the derived
omega_m h^2, shaded with vertical spans (axvspan).
"""

import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")

from emulator import plotting

FAILURES = []


def report(label, ok, detail, aid=None):
  """Print one acceptance line and record a failure."""
  mark = "PASS" if ok else "FAIL"
  print("  [" + mark + "] " + label + "  (" + detail + ")")
  # (queue 2) the per-leg assertion manifest the board folds into this gate's
  # executed set: one reserved '##AID <aid> <result>' line per acceptance leg.
  # The child's exit status stays the single aggregate verdict, not a leg.
  if aid is not None:
    print("##AID " + aid + " " + mark)
  if not ok:
    FAILURES.append(label)


def synthetic_source(n_rows):
  """Build a synthetic LCDM parameter scatter the windows are sharp on.

  Arguments:
    n_rows = number of synthetic cosmologies.

  Returns:
    (source, names, dchi2): source = {"C", "idx"} with columns
    omegab / omegam / h0 / ns in physical ranges that straddle the cut
    bounds; names the column names; dchi2 a positive colour array.
  """
  rng = np.random.default_rng(0)
  omegab = rng.normal(0.048, 0.004, n_rows)
  omegam = rng.normal(0.30, 0.05, n_rows)
  h0 = rng.normal(67.0, 5.0, n_rows)
  ns = rng.normal(0.96, 0.02, n_rows)
  columns = np.column_stack([omegab, omegam, h0, ns]).astype("float64")
  source = {"C": columns, "idx": np.arange(n_rows)}
  names = ["omegab", "omegam", "h0", "ns"]
  dchi2 = rng.uniform(0.01, 1.0, n_rows)
  return source, names, dchi2


def facecolors(ax):
  """Every rgba face colour of the CUT-SHADING artists on one axis.

  The cut shading is exactly the zorder-0 layer: plotting._shade_cuts
  draws every window fill (the 2-D contourf regions and the omh2
  marginal's axvspan bands) at zorder 0, deliberately under the data.
  The data artists — the viridis-coloured scatter of every off-diagonal
  panel, the shared colorbar, the density lines — all live at higher
  zorder, and a scatter IS a filled collection (the original version of
  this check assumed points draw as lines and mis-counted every point
  cloud as an off-grey fill). Filtering by zorder == 0 selects the
  shading layer with no assumption about how the data is drawn.

  Arguments:
    ax = a matplotlib Axes.

  Returns:
    a list of rgba tuples (length-4), one per zorder-0 shading artist.
  """
  colours = []
  for coll in ax.collections:
    if coll.get_zorder() != 0:
      continue
    fc = np.asarray(coll.get_facecolor())
    for row in range(fc.shape[0]):
      colours.append(tuple(fc[row]))
  for patch in ax.patches:
    if patch.get_zorder() != 0:
      continue
    fc = np.asarray(patch.get_facecolor())
    colours.append(tuple(fc))
  return colours


def close_to_grey(rgba):
  """Whether an rgba matches plotting._CUT_GREY within a small tolerance."""
  grey = np.asarray(plotting._CUT_GREY)
  return bool(np.allclose(np.asarray(rgba), grey, atol=1.0e-6))


def main():
  """Render the four-window triangle and assert its shading artists."""
  print("== triangle-shading ==")
  source, names, dchi2 = synthetic_source(n_rows=4000)

  # all four windows active, bounds that straddle the synthetic scatter.
  cuts = {"omegabh2_lo": 0.014,
          "omegabh2_hi": 0.035,
          "omegam2h2_lo": 0.015,
          "omegam2h2_hi": 0.08,
          "omegamh2_lo": 0.10,
          "omegamh2_hi": 0.18,
          "omegamh2ns_lo": 0.10,
          "omegamh2ns_hi": 0.17}

  fig = plotting._lcdm_triangle_fig(source=source,
                                    names=names,
                                    dchi2=dchi2,
                                    cuts=cuts)
  report("the triangle figure was produced",
         fig is not None,
         "fig is None means < 2 LCDM columns recognized",
         aid="triangle-shading.figure-produced")
  if fig is None:
    print("\ntriangle-shading: 1 FAILURE(S)")
    return 1

  # walk the axes: count grey fills, off-grey artists, and axvspan bands.
  shaded_panels = 0
  off_grey = 0
  span_bands = 0
  for ax in fig.axes:
    colours = facecolors(ax)
    grey_here = 0
    for rgba in colours:
      if close_to_grey(rgba):
        grey_here = grey_here + 1
      elif rgba[3] > 0.0:
        # a coloured, non-transparent artist ON THE ZORDER-0 SHADING
        # LAYER that is not the cut grey: a real shading-colour bug
        # (facecolors already excluded the data artists by zorder).
        off_grey = off_grey + 1
    if grey_here > 0:
      shaded_panels = shaded_panels + 1
    # count only the zorder-0 patches (the axvspan cut bands); a patch
    # at higher zorder belongs to the data/decoration, not the shading.
    for patch in ax.patches:
      if patch.get_zorder() == 0:
        span_bands = span_bands + 1

  print("shaded panels: " + str(shaded_panels)
        + " ; axvspan patches (omh2 band): " + str(span_bands)
        + " ; off-grey fills: " + str(off_grey))

  report("grey fills appear on the 2-D panels",
         shaded_panels > 0,
         "shaded panels " + str(shaded_panels),
         aid="triangle-shading.shading-layer-present")
  report("every cut fill uses the one shared rgba (_CUT_GREY)",
         off_grey == 0,
         "off-grey filled artists " + str(off_grey),
         aid="triangle-shading.all-shading-fills-use-shared-gray")
  report("the omh2 1-D marginal carries an axvspan band",
         span_bands > 0,
         "axvspan patches " + str(span_bands),
         aid="triangle-shading.zorder-zero-span-present")

  print("")
  if len(FAILURES) == 0:
    print("triangle-shading: ALL PASS")
    return 0
  print("triangle-shading: " + str(len(FAILURES)) + " FAILURE(S)")
  return 1


if __name__ == "__main__":
  sys.exit(main())
