"""Training-history, learning-curve, coverage, xi, and family plots.

This module draws every matplotlib figure the package produces, all in
a colorblind-safe palette (never red with green). plot_history draws
the training history, plot_diagnostics the multipage diagnostics PDF
(history, coverage, the local-linear floor, the hard-direction
regression, a chi2-colored lcdm triangle, and the ln-parameter PCA
plane colored by chi2 and by training sparsity; a CMB / scalar /
background run appends its family's pages — per-multipole residual
bands and short-period wiggle content for CMB, per-output residual
pages for scalars, per-redshift bands plus the derived-distance page
for the background). A triangle plot is the grid showing every
parameter pair as a scatter panel with each parameter's own 1-D
distribution on the diagonal; PCA is principal component analysis,
the rotation of the parameter axes onto the directions that carry
the most variance.
plot_learning_curves overlays f(delta-chi2 > thr) vs N_train curves
(the sweep / bake-off output). source_param_samples, dv_to_xi, and
plot_xi handle the parameter-coverage triangle and the xi
correlation-function curves. The "_"-prefixed helpers draw the
individual panels the public functions share.

PS: whitened = rotated into the covariance eigenbasis and scaled to
unit variance, the decorrelated space the chi2 residuals live in;
dump = the full on-disk array from the data-generation run, one row
per cosmology (the dv dump is the .npy, the param dump the .txt).
"""

import itertools
import warnings
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from getdist import MCSamples
from getdist import plots as gdplots

# colorblind-safe palette, no red/green (Wong 2011 minus its green
# and vermillion): blue, orange, reddish-purple, black, sky-blue.
_CB = ["#0072B2", "#E69F00", "#CC79A7", "#000000", "#56B4E9"]


def _finish(fig, savepath):
  """Save the figure and close it, or show it interactively.

  Arguments:
    fig      = the matplotlib figure to finish.
    savepath = the output path (format from the extension, e.g. .pdf);
               the figure is written there and closed, because a batch
               script has no display. None shows it interactively
               instead.
  """
  if savepath is not None:
    fig.savefig(savepath, bbox_inches="tight")
    plt.close(fig)
  else:
    plt.show()


def _history_panels(ax_loss, ax_frac, train_losses, medians,
                    means, fracs, thresholds):
  """
  Draw the two training-history panels.

  ax_loss: train loss / val median / val mean vs epoch (log y, as
  the mean is heavy-tailed far above the median). ax_frac: fraction
  of val points over each delta-chi2 threshold vs epoch. Shared by
  plot_history (1x2) and plot_diagnostics (2x2) so the two never
  drift apart.

  Arguments:
    ax_loss, ax_frac = the two axes to draw on.
    train_losses     = per-epoch training loss (list).
    medians / means  = per-epoch val median / mean chi2 (lists).
    fracs            = per-epoch list of 1D tensors; fracs[i] is
                       the fraction over each threshold at epoch i+1.
    thresholds       = 1D tensor of delta-chi2 cutoffs (labels).
  """
  epochs = range(1, len(medians) + 1)
  # x = epoch, y = each per-epoch curve: train loss, val median,
  # val mean.
  ax_loss.semilogy(epochs,
                   train_losses,
                   color=_CB[0],
                   label="train")
  ax_loss.semilogy(epochs,
                   medians,
                   color=_CB[1],
                   label="val median")
  ax_loss.semilogy(epochs,
                   means,
                   color=_CB[2],
                   label="val mean")
  ax_loss.set_xlabel("epoch")
  ax_loss.set_ylabel("loss")
  ax_loss.legend(frameon=False)

  fr = torch.stack(fracs).cpu()      # (nepochs, n_thr)
  for j, t in enumerate(thresholds.tolist()):
    # x = epochs, y = fraction of val points over threshold j.
    ax_frac.plot(epochs,
                 fr[:, j],
                 color=_CB[j % len(_CB)],
                 label=f"> {t:g}")
  ax_frac.set_xlabel("epoch")
  ax_frac.set_ylabel("fraction of val points")
  ax_frac.legend(frameon=False, title="delta chi2")


# the chi2 color band in log10: floor and saturation shared by every
# chi2-colored scatter, and the pinned colorbar limits (pinning keeps
# the colorbars identical across runs and pages, so shades compare).
_CHI2_CBAND = (-2.0, 1.5)


def _log_dchi2_color(dchi2, lo=_CHI2_CBAND[0], hi=_CHI2_CBAND[1]):
  """
  log10 delta-chi2 clipped to a fixed color band.

  The raw chi2 spans many decades and a handful of catastrophic
  outliers would set the color scale, washing every ordinary point
  into one dark shade. Clipping to [lo, hi] (default delta-chi2 in
  [0.01, ~30]) spends the whole colormap on the decision-relevant
  band around the 0.2 goal; anything beyond saturates at the end
  color. The floor also guards log10 against a numerically zero
  chi2.

  Arguments:
    dchi2 = per-point delta-chi2 values.
    lo    = lower color bound in log10 (default -2, chi2 = 0.01).
    hi    = upper color bound in log10 (default +1.5, chi2 ~ 30).

  Returns:
    (N,) clipped log10 values, ready to be a scatter color.
  """
  c = np.asarray(dchi2, dtype="float64")
  return np.clip(np.log10(np.maximum(c, 1e-12)), lo, hi)


def _coverage_panels(ax_scatter, ax_hist, knn_dist, dchi2, k_nn):
  """
  Draw the two coverage-diagnostic panels.

  ax_scatter: per-val hardness log10(dchi2) vs local sparsity (mean
  distance to the k nearest training points), with the 0.2 goal
  line. ax_hist: sparsity distributions of the good (dchi2<=0.2) and
  bad (dchi2>0.2) populations, a right-shifted "bad" histogram
  means failures live where training is scarce.

  Arguments:
    ax_scatter, ax_hist = the two axes to draw on.
    knn_dist = (Nval,) mean distance to the k nearest train points.
    dchi2    = (Nval,) per-val delta-chi2 (same row order).
    k_nn     = the k used (for the axis labels).
  """
  y   = np.log10(np.maximum(dchi2, 1e-4))
  bad = dchi2 > 0.2

  # (a) hardness vs local sparsity. x = knn_dist, y = log10 dchi2
  # (full range, so outliers stay visible as points); color = the
  # clipped log10 (saturated ends, so the bulk keeps color
  # resolution). Points draw dark-to-bright so the rare hard ones
  # sit on top of the overplotted bulk instead of being buried.
  # The dashed line is the 0.2 goal.
  c = _log_dchi2_color(dchi2)
  order = np.argsort(c)
  sc = ax_scatter.scatter(knn_dist[order], y[order], s=5,
                          c=c[order],
                          cmap="viridis",
                          vmin=_CHI2_CBAND[0],
                          vmax=_CHI2_CBAND[1])
  ax_scatter.axhline(np.log10(0.2), color="0.4", lw=1, ls="--")
  ax_scatter.set_xlabel(f"mean dist to {k_nn} nearest train pts")
  ax_scatter.set_ylabel(r"$\log_{10}\,\Delta\chi^2$")
  # ax.figure is the parent figure; add the colorbar to it. The
  # extend arrows mark that values continue past the clipped ends.
  ax_scatter.figure.colorbar(sc,
                             ax=ax_scatter,
                             extend="both",
                             label=r"$\log_{10}\,\Delta\chi^2$")

  # (b) good vs bad sparsity. x = knn_dist, y = density; shared
  # bins so the two histograms are comparable. The range clips the
  # far knn tails (a lone outlier stretches equal-width bins until
  # the narrow population falls into a single bar), and the bin
  # width follows the narrower population (Freedman-Diaconis per
  # population, take the smaller), so a tight "good" peak still
  # shows its shape next to a broad "bad" one.
  lo = np.percentile(knn_dist, 0.5)
  hi = np.percentile(knn_dist, 99.5)
  widths = []
  for m in (~bad, bad):
    if m.sum() >= 2:
      e = np.histogram_bin_edges(knn_dist[m], bins="fd")
      if len(e) > 1 and e[1] > e[0]:
        widths.append(e[1] - e[0])
  width = min(widths) if widths else (hi - lo) / 40.0
  nbins = int(np.ceil((hi - lo) / max(width, 1e-12)))
  nbins = min(max(nbins, 20), 150)   # readable floor / ceiling
  bins = np.linspace(lo, hi, nbins + 1)
  ax_hist.hist(knn_dist[~bad],
               bins=bins,
               density=True,
               alpha=0.6,
               color=_CB[0],
               label="good (dchi2<0.2)")
  ax_hist.hist(knn_dist[bad],
               bins=bins,
               density=True,
               alpha=0.6,
               color=_CB[1],
               label="bad (dchi2>0.2)")
  ax_hist.set_xlabel(f"mean dist to {k_nn} nearest train pts")
  ax_hist.set_ylabel("density")
  ax_hist.legend(frameon=False)


def plot_history(train_losses,
                 medians,
                 means,
                 fracs,
                 thresholds,
                 savepath=None):
  """Plot a run_emulator training history (the two history panels).

  Left: train loss, val median, val mean vs epoch (log y). Right:
  fraction of val points over each delta-chi2 threshold vs epoch.
  The first four arguments are the histories run_emulator returns.

  Arguments:
    train_losses = per-epoch training loss (list of floats); the
                   sqrt-trimmed objective, on a different scale than
                   the raw-chi2 val metrics.
    medians      = per-epoch val median chi2 (list).
    means        = per-epoch val mean chi2 (list).
    fracs        = per-epoch list of 1D tensors; fracs[i] holds the
                   fraction of val points over each threshold at
                   epoch i+1.
    thresholds   = 1D tensor of delta-chi2 cutoffs used in
                   training; labels the right panel.
    savepath     = if given, write the figure there and close; if
                   None (default), show it interactively.
  """
  fig, ax = plt.subplots(1, 2, figsize=(11, 4))
  _history_panels(ax[0], ax[1], train_losses, medians, means,
                  fracs, thresholds)
  fig.tight_layout()
  _finish(fig, savepath)


def plot_learning_curves(curves,
                         threshold=0.2,
                         target=0.10,
                         savepath=None):
  """
  Overlay one or more learning curves: f(delta-chi2 > threshold) vs
  N_train.

  One descending curve per entry, on log-log axes (spreading out the
  small-N regime where methods separate). A curve still falling at the
  largest N is data-limited (more data helps); a flat tail is
  capacity / architecture-limited. Lines use a colorblind palette + a
  marker cycle (no red/green); a single-config sweep passes a
  one-entry dict, a bake-off one entry per variant.

  Arguments:
    curves    = mapping label -> the curve, where the curve is either a
                {N_train: frac} dict or an (sizes, fracs) pair (both
                sorted by N here). label is the legend text.
    threshold = the delta-chi2 cutoff the fraction counts (default 0.2,
                the emulator goal); labels the y axis.
    target    = a horizontal guide at the target fraction (default 0.10);
                None to omit it.
    savepath  = if given, write the figure there and close; else show.
  """
  # marker cycle so overlaid curves stay distinguishable in print.
  markers = ["o", "D", "^", "s", "v", "P"]
  fig, ax = plt.subplots(figsize=(6.8, 5.6))

  for k, (label, curve) in enumerate(curves.items()):
    # accept {N: frac} or a (sizes, fracs) pair.
    if isinstance(curve, dict):
      keys  = sorted(curve)
      sizes = np.array(keys, dtype="float64")
      fvals = []
      for n in keys:
        fvals.append(curve[n])
      fracs = np.array(fvals, dtype="float64")
    else:
      sizes = np.asarray(curve[0], dtype="float64")
      fracs = np.asarray(curve[1], dtype="float64")
      order = np.argsort(sizes)            # plot left-to-right in N
      sizes, fracs = sizes[order], fracs[order]
    # x = N_train, y = fraction over the threshold.
    ax.plot(sizes,
            fracs,
            "-" + markers[k % len(markers)],
            color=_CB[k % len(_CB)],
            lw=2.5,
            ms=8,
            label=label)

  ax.set_xscale("log")
  ax.set_yscale("log")
  ax.set_xlabel(r"$N_{\rm train}$")
  ax.set_ylabel(rf"$f(\Delta\chi^2 > {threshold:g})$")
  if target is not None:
    ax.axhline(target, color="0.6", ls="--", lw=1,
               label=f"target {target:g}")
  ax.legend(frameon=False)
  fig.tight_layout()
  _finish(fig, savepath)


def plot_sweep_curve(param,
                     values,
                     fracs,
                     threshold=0.2,
                     target=0.10,
                     design_label=None,
                     savepath=None):
  """
  One-hyperparameter sweep figure: f(delta-chi2 > threshold) vs the
  swept values.

  The generic twin of plot_learning_curves for an arbitrary knob.
  Numeric values draw a connected curve (log x when the values are
  all positive and span more than a factor 20, so a log-spaced lr
  grid reads evenly); categorical values (activation names, a
  film True/False pair) draw one marker per value with the labels
  as x ticks. y is logarithmic when every fraction is positive,
  linear otherwise (a perfect 0.0 point would break a log axis).

  Arguments:
    param     = the swept hyperparameter's dotted YAML path (the x
                label).
    values    = the swept values, in sweep order (numbers, strings,
                or booleans).
    fracs     = per-value fractions aligned with `values`.
    threshold = the delta-chi2 cutoff the fraction counts (labels
                the y axis).
    target    = a horizontal guide at the target fraction (default
                0.10); None to omit it.
    design_label = resolved model and activation facts shown as the title;
                   None omits the title.
    savepath  = if given, write the figure there and close; else
                show.
  """
  # booleans are ints in Python; label them instead of plotting 0/1.
  numeric = True
  for v in values:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
      numeric = False
  fr = np.asarray(fracs, dtype="float64")

  fig, ax = plt.subplots(figsize=(6.8, 5.6))
  if numeric:
    xs = np.asarray(values, dtype="float64")
    order = np.argsort(xs)                # draw left-to-right
    # x = the swept values, y = fraction over the threshold.
    ax.plot(xs[order],
            fr[order],
            "-o",
            color=_CB[0],
            lw=2.5,
            ms=8)
    if xs.min() > 0 and xs.max() / xs.min() > 20:
      ax.set_xscale("log")
  else:
    xs = np.arange(len(values))
    # x = value index (ticks carry the labels), y = the fraction.
    ax.plot(xs,
            fr,
            "o",
            color=_CB[0],
            ms=10)
    labels = []
    for v in values:
      labels.append(str(v))
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right")
  if np.all(fr[np.isfinite(fr)] > 0):
    ax.set_yscale("log")
  ax.set_xlabel(param)
  ax.set_ylabel(rf"$f(\Delta\chi^2 > {threshold:g})$")
  if design_label is not None:
    ax.set_title(design_label)
  if target is not None:
    ax.axhline(target, color="0.6", ls="--", lw=1,
               label=f"target {target:g}")
    ax.legend(frameon=False)
  fig.tight_layout()
  _finish(fig, savepath)


def _floor_panel(ax, floor):
  """
  Draw the local-linear data-floor panel.

  Per val point: the model's delta-chi2 vs the data-only floor (the
  local-linear prediction's delta-chi2), log-log. Points on the
  diagonal mean the net is at what a smooth local method extracts
  from the data (data-limited); points well above mean it has
  headroom. Dotted lines mark the 0.2 goal on each axis.

  Arguments:
    ax    = the axis to draw on.
    floor = the dict local_linear_floor returned (dchi2_floor,
            dchi2_model, f_floor, f_model, f_hard).
  """
  lo = 1e-3
  dchi2_floor = floor["dchi2_floor"]
  dchi2_model = floor["dchi2_model"]
  # x = data-only floor dchi2, y = model dchi2; clip both at lo so
  # a near-zero point stays on the log axes.
  ax.scatter(np.maximum(dchi2_floor, lo),
             np.maximum(dchi2_model, lo),
             s=5, alpha=0.4, color=_CB[0])
  mx = max(dchi2_floor.max(), dchi2_model.max())
  ax.plot([lo, mx], [lo, mx], "k--", lw=1)
  ax.set_xscale("log")
  ax.set_yscale("log")
  ax.axhline(0.2, color="0.6", lw=1, ls=":")
  ax.axvline(0.2, color="0.6", lw=1, ls=":")
  ax.set_xlabel(r"data-only $\Delta\chi^2$ (local-linear floor)")
  ax.set_ylabel(r"model $\Delta\chi^2$")
  ax.set_title(f"f_model {floor['f_model']:.3f}  vs  "
               f"f_floor {floor['f_floor']:.3f}  "
               f"(pure hardness {floor['f_hard']:.3f})")


def _hard_direction_panels(ax_uni, ax_joint, hd):
  """
  Draw the hard-direction regression as two bar charts.

  ax_uni: each feature's univariate |correlation| with log10 dchi2
  (a collinearity-robust ranking). ax_joint: the joint log-linear
  OLS coefficients (the alpha, beta, ... combination). The joint R^2
  and the ln(omega_b h^2)-alone R^2 are in the titles. Both panels
  share the feature order (descending univariate |corr|).

  Arguments:
    ax_uni, ax_joint = the two axes to draw on.
    hd = the dict hard_direction_regression returned (labels,
         univariate, joint_coef, r2, r2_omega).
  """
  labels = hd["labels"]
  uni    = hd["univariate"]
  coef   = hd["joint_coef"]
  # order features by descending univariate |corr|; both panels
  # share this order so the bars line up.
  order = np.argsort(np.abs(uni))[::-1]
  ypos  = np.arange(len(order))
  names = []
  for j in order:
    names.append(labels[j])

  # barh(y, width): y = bar slot, width = value.
  ax_uni.barh(ypos, np.abs(uni)[order], color=_CB[0])
  ax_uni.set_yticks(ypos)
  ax_uni.set_yticklabels(names)
  ax_uni.invert_yaxis()                 # strongest feature at top
  ax_uni.set_xlabel("univariate |corr| with log10 dchi2")
  ax_uni.set_title("univariate ranking")

  ax_joint.barh(ypos, coef[order], color=_CB[1])
  ax_joint.set_yticks(ypos)
  ax_joint.set_yticklabels(names)
  ax_joint.invert_yaxis()
  ax_joint.axvline(0.0, color="0.6", lw=1)
  ax_joint.set_xlabel("joint log-linear coefficient")
  ax_joint.set_title(f"joint R2 {hd['r2']:.3f}  |  "
                     f"ln(omega_b h2) alone {hd['r2_omega']:.3f}")


def _save_pages(figs, savepath):
  """
  Save figures as a multipage PDF, or show them.

  If savepath is given, write every figure as one page of a single
  PDF (matplotlib's PdfPages) and close them, a batch script has
  no display; if None, show them interactively.

  Arguments:
    figs     = list of matplotlib Figures, one per page.
    savepath = the .pdf path, or None to show.
  """
  if savepath is None:
    plt.show()
    return
  from matplotlib.backends.backend_pdf import PdfPages
  with PdfPages(savepath) as pdf:
    for f in figs:
      pdf.savefig(f, bbox_inches="tight")
      plt.close(f)


# The basic lcdm subset for the chi2-colored triangle: lowercased dump
# names seen in covmat headers, each with its getdist LaTeX label (no
# surrounding $). tau is not sampled in these dumps; w0 / wa and the
# nuisances are deliberately left out (the triangle reads best small).
_LCDM_ALIASES = (
  ("as",      r"A_\mathrm{s}"),
  ("as_1e9",  r"10^9 A_\mathrm{s}"),
  ("logas",   r"\ln(10^{10} A_\mathrm{s})"),
  ("loga",    r"\ln(10^{10} A_\mathrm{s})"),
  ("ns",      r"n_\mathrm{s}"),
  ("n_s",     r"n_\mathrm{s}"),
  ("h0",      r"H_0"),
  ("omegab",  r"\Omega_\mathrm{b}"),
  ("omega_b", r"\Omega_\mathrm{b}"),
  ("omegam",  r"\Omega_\mathrm{m}"),
  ("omega_m", r"\Omega_\mathrm{m}"),
)


def _lcdm_columns(names):
  """
  The lcdm subset of a dump's parameter names, with LaTeX labels.

  Matches each name (case-insensitively) against _LCDM_ALIASES and
  returns the ones present, in the alias table's canonical order
  (A_s, n_s, H0, Omega_b, Omega_m). Names the table does not know
  (w0, wa, photo-z shifts, IA amplitudes) are skipped.

  Arguments:
    names = parameter column names, in the dump's column order.

  Returns:
    (kept, labels): the matched dump names and their labels.
  """
  kept   = []
  labels = []
  for alias, label in _LCDM_ALIASES:
    for n in names:
      if n.lower() == alias and n not in kept:
        kept.append(n)
        labels.append(label)
  return kept, labels


# which lowercased dump names play each role in the physical cuts
# (plus our own derived omegamh2 column); used to find the triangle
# panels whose two axes determine a cut variable.
_CUT_ROLES = (
  ("h0",   ("h0",)),
  ("ob",   ("omegab", "omega_b")),
  ("om",   ("omegam", "omega_m")),
  ("omh2", ("omegamh2",)),
  ("ns",   ("ns",)),
)

# same semi-transparent grey for every window; separate fills alpha-stack,
# so a double overlap darkens visibly and the union's outer edge traces the
# allowed region, while a triple still sits clearly under the points.
_CUT_GREY = (0.55, 0.55, 0.55, 0.30)


def _cut_role(name):
  """Classify one parameter column's role in the physical cuts.

  The triangle plot marks the physical-cut windows only on the panels
  whose two axes make a window sharp; that decision needs to know which
  column is H0, which is Omega_b, and so on, whatever alias the dump
  used for it.

  Arguments:
    name = the column name as the parameter table declares it.

  Returns:
    the role string ("h0" / "ob" / "om" / "omh2"), or None when the
    column plays no part in the physical cuts.
  """
  low = name.lower()
  for role, aliases in _CUT_ROLES:
    if low in aliases:
      return role
  return None


def _window_masks(rx, ry, xx, yy, cuts):
  """
  Per-window physical-cut exclusion masks sharp on one triangle panel.

  Returns one boolean grid per active window that this panel's two axes
  make sharp (a list of (window_name, mask)); empty when no window is
  sharp here. A window is sharp only where its derived quantity is a
  function of the panel's axes (the "sharp only" rule): every other
  panel merely thins the cloud marginally and gets nothing. The
  formulas mirror phys_cut_idx's quantity table (data_staging.py)
  exactly.

  Sharp panels (the coverage table in
  ai/notes/data-generation-and-cuts.md):
    omegabh2   = Omega_b (H0/100)^2          on (ob, h0)
    omegam2h2  = (Omega_m H0/100)^2          on (om, h0),
               = Omega_m * omegamh2          on (om, omh2),
               = omegamh2^2 / (H0/100)^2     on (h0, omh2)
    omegamh2   = Omega_m (H0/100)^2          on (om, h0),
               = the omh2 axis value         on any panel with an omh2 axis
    omegamh2ns = omegamh2 * n_s              on (ns, omh2)

  Arguments:
    rx, ry = cut roles of the panel's x and y axes (see _cut_role).
    xx, yy = meshgrid of axis values covering the panel.
    cuts   = the validated param_cuts mapping (omegabh2_lo / _hi,
             omegam2h2_lo / _hi, omegamh2_lo / _hi, omegamh2ns_lo /
             _hi; any absent = that side not cut).

  Returns:
    a list of (window_name, boolean_grid), True where that window
    excludes; empty when no window is sharp or active on this panel.
  """
  pair  = {rx, ry}
  masks = []

  # add a window's mask: excluded where q <= lo or q >= hi, matching
  # phys_cut_idx's strict keep window lo < q < hi (an inactive side is
  # skipped; an all-False mask is dropped).
  def add(name, q, lo, hi):
    """Append this window's exclusion mask when either bound is active.

    Arguments:
      name = the window's name, kept beside its mask.
      q    = the derived quantity on the panel grid.
      lo   = the lower bound, or None for no lower cut.
      hi   = the upper bound, or None for no upper cut.
    """
    if lo is None and hi is None:
      return
    bad = np.zeros(q.shape, dtype=bool)
    if lo is not None:
      bad |= q <= lo
    if hi is not None:
      bad |= q >= hi
    if bad.any():
      masks.append((name, bad))

  # omegabh2 = Omega_b (H0/100)^2, sharp on (ob, h0).
  # (phys_cut_idx quantity table: _omega_b_h2.)
  if pair == {"h0", "ob"}:
    h0 = xx if rx == "h0" else yy
    ob = yy if rx == "h0" else xx
    add("omegabh2", ob * (h0 / 100.0) ** 2,
        cuts.get("omegabh2_lo"), cuts.get("omegabh2_hi"))

  # omegam2h2 = (Omega_m H0/100)^2 = Gamma^2, sharp three ways.
  # (phys_cut_idx quantity table: _omega_m2_h2.)
  g2 = None
  if pair == {"h0", "om"}:
    h0 = xx if rx == "h0" else yy
    om = yy if rx == "h0" else xx
    g2 = (om * h0 / 100.0) ** 2
  elif pair == {"om", "omh2"}:
    om   = xx if rx == "om" else yy
    omh2 = yy if rx == "om" else xx
    g2 = om * omh2
  elif pair == {"h0", "omh2"}:
    h0   = xx if rx == "h0" else yy
    omh2 = yy if rx == "h0" else xx
    g2 = omh2 ** 2 / (h0 / 100.0) ** 2
  if g2 is not None:
    add("omegam2h2", g2,
        cuts.get("omegam2h2_lo"), cuts.get("omegam2h2_hi"))

  # omegamh2 = Omega_m (H0/100)^2, sharp on (om, h0); on any panel with
  # an omh2 axis it is that axis value directly, so a 1-D band across it.
  # (phys_cut_idx quantity table: _omega_m_h2.)
  if pair == {"h0", "om"}:
    h0 = xx if rx == "h0" else yy
    om = yy if rx == "h0" else xx
    add("omegamh2", om * (h0 / 100.0) ** 2,
        cuts.get("omegamh2_lo"), cuts.get("omegamh2_hi"))
  elif "omh2" in pair:
    omh2 = xx if rx == "omh2" else yy
    add("omegamh2", omh2,
        cuts.get("omegamh2_lo"), cuts.get("omegamh2_hi"))

  # omegamh2ns = omegamh2 * n_s, sharp on (ns, omh2).
  # (phys_cut_idx quantity table: _omega_m_h2_ns.)
  if pair == {"ns", "omh2"}:
    ns   = xx if rx == "ns" else yy
    omh2 = yy if rx == "ns" else xx
    add("omegamh2ns", ns * omh2,
        cuts.get("omegamh2ns_lo"), cuts.get("omegamh2ns_hi"))

  return masks


def _shade_cuts(g, plot_names, cuts):
  """
  Gray out the physically-cut regions on a getdist triangle.

  Walks the lower-triangle panels; on each, every window the two axes
  make sharp (see _window_masks) fills its excluded region in the same
  semi-transparent grey under the points, so the fills alpha-stack and
  the union's outer edge traces the allowed region. The derived
  omega_m h^2 axis also gets a 1-D exclusion band on its diagonal
  marginal. One caption line names the convention, so an empty corner
  reads as "removed by the cut", not as an emulator failure.

  Arguments:
    g          = the getdist subplot plotter (after triangle_plot).
    plot_names = the plotted column names, in triangle order
                 (panel [i][j] has x = plot_names[j],
                 y = plot_names[i]).
    cuts       = the validated param_cuts mapping (see _window_masks).
  """
  n = len(plot_names)
  # lower-triangle 2-D panels: one grey fill per sharp window, stacked.
  for i in range(1, n):
    for j in range(i):
      ax = g.subplots[i][j]
      if ax is None:
        continue
      rx = _cut_role(plot_names[j])
      ry = _cut_role(plot_names[i])
      if rx is None or ry is None:
        continue
      xs = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 200)
      ys = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 200)
      xx, yy = np.meshgrid(xs, ys)
      masks = _window_masks(rx=rx, ry=ry, xx=xx, yy=yy, cuts=cuts)
      if not masks:
        continue
      for _name, bad in masks:
        # one fill per window, all the same grey; zorder 0 renders them
        # under the already-drawn points, and the separate artists
        # alpha-stack so overlaps darken. The [0.5, 1.5] band selects
        # exactly the True cells.
        ax.contourf(xx, yy, bad.astype(float),
                    levels=[0.5, 1.5], colors=[_CUT_GREY], zorder=0)
      # contourf can widen the limits; pin them back to the data's.
      ax.set_xlim(xs[0], xs[-1])
      ax.set_ylim(ys[0], ys[-1])
  # the omega_m h^2 1-D marginal (a diagonal panel) gets a band too.
  _shade_omh2_marginal(g=g, plot_names=plot_names, cuts=cuts)
  g.fig.text(0.99, 0.99,
             "gray: region removed by the physical cuts",
             ha="right", va="top", fontsize=9, color="0.35")


def _shade_omh2_marginal(g, plot_names, cuts):
  """
  Band out the omega_m h^2 window on its 1-D diagonal marginal.

  The marginal is the 1-D distribution of that parameter alone, all
  others summed over — the diagonal panel of the triangle plot.
  omegamh2 is a derived triangle axis, so its cut is a plain interval
  on the omh2 marginal: axvspan the excluded low / high ends in the
  same grey, drawing nothing when the window is off, the omh2 axis is
  absent, or a bound sits outside the panel range. (phys_cut_idx
  quantity table: _omega_m_h2.)

  Arguments:
    g          = the getdist subplot plotter.
    plot_names = the plotted column names, triangle order.
    cuts       = the validated param_cuts mapping.
  """
  lo = cuts.get("omegamh2_lo")
  hi = cuts.get("omegamh2_hi")
  if (lo is None and hi is None) or "omegamh2" not in plot_names:
    return
  k  = plot_names.index("omegamh2")
  ax = g.subplots[k][k]
  if ax is None:
    return
  x0, x1 = ax.get_xlim()
  # excluded omh2 < lo and omh2 > hi, each drawn only where it lands
  # inside the panel's x range (a bound outside it shades nothing).
  if lo is not None and lo > x0:
    ax.axvspan(x0, min(lo, x1), color=_CUT_GREY, zorder=0)
  if hi is not None and hi < x1:
    ax.axvspan(max(hi, x0), x1, color=_CUT_GREY, zorder=0)
  ax.set_xlim(x0, x1)


def _lcdm_triangle_fig(source, names, dchi2, cuts=None):
  """
  getdist triangle of a source's lcdm parameters, colored by chi2.

  A triangle plot shows every parameter pair as one scatter panel,
  with each parameter's own 1-D distribution on the diagonal; getdist
  draws it. Each off-diagonal panel is a scatter of the source's cosmologies
  (one point per used row, in the same sorted-idx order
  eval_source_chi2 scores), colored by log10 delta-chi2; the
  diagonal shows the 1D densities. It answers where in lcdm space
  the emulator fails, not just how often. When both Omega_m and H0
  are present, the derived omega_m h^2 = Omega_m (H0/100)^2 (the
  structure-amplitude direction) is appended as an extra triangle
  axis. Returns None when fewer than two lcdm columns are
  recognized in `names`.

  Arguments:
    source = source dict with "C" (param dump) and "idx" (used rows).
    names  = parameter column names, in the dump's column order.
    dchi2  = (N,) per-row delta-chi2, sorted-idx order (as returned
             by coverage_diagnostic / eval_source_chi2).
    cuts   = optional validated param_cuts mapping (omegabh2_lo /
             _hi, omegam2h2_lo / _hi, omegamh2_lo / _hi, omegamh2ns_lo
             / _hi); when given, each window's excluded region is
             shaded grey on the panels it makes sharp (see
             _shade_cuts / _window_masks), so an empty corner is not
             mistaken for an emulator failure.

  Returns:
    the matplotlib Figure of the triangle, or None.
  """
  lcdm, labels = _lcdm_columns(names)
  if len(lcdm) < 2:
    return None

  rows = np.sort(source["idx"])
  # raw physical parameters of the used rows (never whitened).
  P = np.asarray(source["C"][rows], dtype="float64")
  cols = []
  for n in lcdm:
    cols.append(names.index(n))

  # color = clipped log10 delta-chi2 (saturated ends so outliers do
  # not wash out the bulk; see _log_dchi2_color for the band).
  logc = _log_dchi2_color(dchi2)

  # derived omega_m h^2 = Omega_m * (H0 / 100)^2, added as its own
  # triangle axis when both parents are among the matched columns
  # (a derived column appended before MCSamples is built).
  vals = P[:, cols]
  i_h0 = None
  i_om = None
  for i, n in enumerate(lcdm):
    if n.lower() == "h0":
      i_h0 = i
    if n.lower() in ("omegam", "omega_m"):
      i_om = i
  plot_names = list(lcdm)
  all_labels = list(labels)
  columns    = [vals]
  if i_h0 is not None and i_om is not None:
    omh2 = vals[:, i_om] * (vals[:, i_h0] / 100.0) ** 2
    columns.append(omh2[:, None])
    plot_names.append("omegamh2")
    all_labels.append(r"\Omega_\mathrm{m} h^2")

  # one extra MCSamples column holds the color, so getdist's
  # plot_3d_with_param can read it by name.
  columns.append(logc[:, None])
  data = np.concatenate(columns, axis=1)
  samples = MCSamples(samples=data,
                      names=plot_names + ["logdchi2"],
                      labels=all_labels + [r"\log_{10}\Delta\chi^2"],
                      settings={"smooth_scale_1D": 0.3,
                                "smooth_scale_2D": 0.3,
                                "fine_bins_2D": 512})

  g = gdplots.get_subplot_plotter(width_inch=9)
  # viridis: sequential and colorblind-safe (the palette rule).
  g.settings.colormap_scatter = "viridis"
  # plot_3d_with_param turns every off-diagonal panel into a scatter
  # colored by that column, with one shared colorbar.
  g.triangle_plot(samples,
                  plot_names,
                  plot_3d_with_param="logdchi2")
  # shade the physically-cut regions so their emptiness reads as
  # "removed by the cut", not as a failure of the emulator.
  if cuts:
    _shade_cuts(g=g, plot_names=plot_names, cuts=cuts)
  return g.fig


def _monomial(vec, labels):
  """
  The product-of-powers string for one ln-space direction.

  A direction with exponent vector v in ln-parameter space is the
  monomial exp(direction) = prod_i p_i^(v_i); the string spells it
  out with the exponents sorted by size and rescaled so the largest
  is 1 (a direction's overall normalization is arbitrary).

  Arguments:
    vec    = (n_params,) monomial exponents of the direction.
    labels = LaTeX labels of the parameters (no surrounding $).

  Returns:
    the mathtext string, e.g. "$\\propto \\ln[H_0^{+1.00}\\,...]$".
  """
  w = vec / np.abs(vec).max()
  order = np.argsort(-np.abs(w))
  terms = []
  for i in order:
    terms.append(f"{labels[i]}^{{{w[i]:+.2f}}}")
  return r"$\propto \ln[" + r"\,".join(terms) + r"]$"


def _pc_label(k, vec, frac, labels):
  """
  Axis label for one ln-parameter principal component.

  The "PCk (share of ln-var)" prefix plus the component's monomial
  (see _monomial).

  Arguments:
    k      = component number (1-based, the "PC1" prefix).
    vec    = (n_params,) monomial exponents of the component (for a
             standardized PCA, eigenvector over the per-parameter
             sigma).
    frac   = this component's share of the total (standardized)
             ln-variance.
    labels = LaTeX labels of the parameters (no surrounding $).

  Returns:
    the axis-label string (matplotlib mathtext).
  """
  return (f"PC{k} ({100.0 * frac:.0f}% of ln-var)  "
          + _monomial(vec, labels))


def _lnparam_pca_fig(source, names, color, clabel, title,
                     fit_target=None, vmin=None, vmax=None):
  """
  First two ln-parameter principal components, colored per row.

  Principal component analysis (PCA) rotates a cloud of points onto
  new axes ordered by how much of the cloud's variance each carries;
  the first two components span the plane the cloud spreads widest
  in. Each ln parameter is centered and scaled to unit variance, and the
  PCA eigendecomposes their sample correlation matrix. The
  standardization is essential: the raw ln variances differ wildly
  (the As prior spans a factor of ten, ns a few percent), so a
  covariance PCA reads back the per-parameter prior widths (PC1 =
  pure As) instead of any correlated combination. On standardized
  variables a component PC = sum_i w_i (ln p_i - mu_i) / sigma_i is
  still a product of parameter powers,
    exp(PC) proportional to  As^(w_As/sigma_As) * ns^(w_ns/sigma_ns) * ...,
  so the axis labels report the effective exponents w_i / sigma_i
  (omega_m h^2 is one such monomial, so the base columns already
  span it; the derived column stays out, keeping the matrix
  non-singular).

  The scatter colors each row by `color` (clipped log10 delta-chi2
  for the hardness page; local training sparsity for the coverage
  page): a color gradient along a direction names the power-law
  combination the colored quantity grows along. With `fit_target`,
  that direction is also measured: a least-squares fit of
  ln(fit_target) on the standardized ln parameters, annotated under
  the title as a monomial with its R^2 (low R^2 = the quantity is
  diffuse, not directional). Returns None when fewer than two LCDM
  columns are recognized, or when a parameter is not strictly
  positive (ln undefined).

  Arguments:
    source     = source dict with "C" (param dump) and "idx" (used
                 rows).
    names      = parameter column names, in the dump's column order.
    color      = (N,) per-row scatter colors, sorted-idx order,
                 already transformed / clipped for display.
    clabel     = colorbar label (mathtext allowed).
    title      = axes title.
    fit_target = optional (N,) positive per-row values; fits
                 ln(fit_target) on the standardized ln parameters
                 and annotates the fitted monomial + R^2.
    vmin, vmax = optional pinned colorbar limits; None lets
                 matplotlib scale to the data (pin the chi2 page to
                 _CHI2_CBAND so colorbars compare across runs).

  Returns:
    the matplotlib Figure, or None.
  """
  lcdm, labels = _lcdm_columns(names)
  if len(lcdm) < 2:
    return None

  rows = np.sort(source["idx"])
  # raw physical parameters of the used rows (never whitened).
  P = np.asarray(source["C"][rows], dtype="float64")
  cols = []
  for n in lcdm:
    cols.append(names.index(n))
  V = P[:, cols]
  if np.any(V <= 0.0):
    return None

  # center each ln parameter and scale it to unit variance (the
  # per-parameter normalization: subtracting the mean is the
  # ln-space form of dividing by a central value; dividing by sigma
  # puts wide and narrow priors on the same footing).
  L   = np.log(V)
  X   = L - L.mean(axis=0)
  sig = X.std(axis=0)
  Z   = X / sig
  # sample correlation matrix of the ln parameters (the covariance
  # of the standardized variables).
  S = np.cov(Z, rowvar=False)

  # eigh returns ascending eigenvalues; flip to descending so
  # column j of evecs is the j-th principal direction.
  evals, evecs = np.linalg.eigh(S)
  order = np.argsort(evals)[::-1]
  evals = evals[order]
  evecs = evecs[:, order]

  # effective monomial exponents: PC = sum_i w_i (ln p_i - mu)/sig_i
  # means exp(PC) carries p_i to the power w_i / sig_i. One column
  # per plotted component. The [:, None] adds a size-1 axis so sig
  # ((k,)) divides each column of evecs ((k, 2)) elementwise.
  expo = evecs[:, :2] / sig[:, None]

  # deterministic sign: make the largest-|exponent| positive (an
  # eigenvector's sign is arbitrary; this fixes the label). Flip the
  # eigenvector and its exponents together so points and label match.
  for j in range(2):
    imax = np.argmax(np.abs(expo[:, j]))
    if expo[imax, j] < 0.0:
      evecs[:, j] = -evecs[:, j]
      expo[:, j]  = -expo[:, j]

  # project the standardized ln parameters on the first two
  # directions.
  pcs  = Z @ evecs[:, :2]
  frac = evals / evals.sum()

  # optional direction fit: ln(fit_target) = a + Z @ b. The fitted b
  # over sigma gives the monomial the quantity grows along; R^2 says
  # how directional (vs diffuse) it is.
  fitline = None
  if fit_target is not None:
    t = np.log(np.maximum(np.asarray(fit_target, dtype="float64"),
                          1e-300))
    A = np.column_stack([np.ones(Z.shape[0]), Z])
    coefs = np.linalg.lstsq(A, t, rcond=None)[0]
    pred = A @ coefs
    ss_res = ((t - pred) ** 2).sum()
    ss_tot = ((t - t.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / max(ss_tot, 1e-300)
    fitline = ("grows along " + _monomial(coefs[1:] / sig, labels)
               + f"  ($R^2$ = {r2:.2f})")

  fig, ax = plt.subplots(figsize=(8, 6))
  # draw dark-to-bright so the rare high-color points sit on top of
  # the overplotted bulk instead of being buried under it.
  cvals = np.asarray(color)
  order = np.argsort(cvals)
  sc = ax.scatter(pcs[order, 0],    # x = PC1 projection
                  pcs[order, 1],    # y = PC2 projection
                  c=cvals[order],
                  cmap="viridis",   # sequential, colorblind-safe
                  s=4,
                  vmin=vmin,
                  vmax=vmax)
  # extend arrows: values continue past the clipped color ends.
  fig.colorbar(sc, ax=ax, extend="both", label=clabel)
  ax.set_xlabel(_pc_label(k=1,
                          vec=expo[:, 0],
                          frac=frac[0],
                          labels=labels))
  ax.set_ylabel(_pc_label(k=2,
                          vec=expo[:, 1],
                          frac=frac[1],
                          labels=labels))
  if fitline is None:
    ax.set_title(title)
  else:
    # two-line title: the page name, then the fitted direction.
    ax.set_title(title + "\n" + fitline, fontsize=11)
  fig.tight_layout()
  return fig


def _cmb_pages(cmb):
  """Build the two CMB-family pages from cmb_residual_diagnostic.

  Page A (2x2): the per-multipole residual bands, fractionally (top
  left; readable for tt/ee/pp, spiky where te crosses zero) and in
  cosmic-variance error-bar units (top right; always well-defined — for
  te read this one), plus the worst-cosmology overlay: predicted vs
  true C_ell (bottom left) and its per-multipole residual/sigma (bottom
  right).

  Page B: the roughness companion — the median absolute short-period
  remainder of the whitened residual vs multipole (the wiggle spectrum
  the roughness term penalizes), with the acoustic band (~200-300 in
  period) noted so over-smoothing or ringing reads at a glance.

  Arguments:
    cmb = the dict cmb_residual_diagnostic returned.

  Returns:
    a list of matplotlib figures.
  """
  ell  = np.asarray(cmb["ell"], dtype="float64")
  spec = str(cmb["spectrum"]).upper()
  figs = []

  fa, ax = plt.subplots(2, 2, figsize=(13, 9))
  # top left: fractional residual bands.
  a = ax[0, 0]
  a.fill_between(ell, cmb["frac_lo95"], cmb["frac_hi95"],
                 color=_CB[4], alpha=0.35, label="95%")
  a.fill_between(ell, cmb["frac_lo68"], cmb["frac_hi68"],
                 color=_CB[0], alpha=0.45, label="68%")
  a.plot(ell, cmb["frac_med"], color=_CB[3], lw=1.0, label="median")
  a.axhline(0.0, color=_CB[3], lw=0.5, ls=":")
  a.set_xlabel(r"$\ell$")
  a.set_ylabel(r"$(\hat C_\ell - C_\ell)\,/\,C_\ell$")
  a.set_title(f"{spec}: fractional residual over the val set")
  a.legend(fontsize=8)
  # top right: residual in error-bar units.
  a = ax[0, 1]
  a.fill_between(ell, cmb["sig_lo95"], cmb["sig_hi95"],
                 color=_CB[4], alpha=0.35, label="95%")
  a.fill_between(ell, cmb["sig_lo68"], cmb["sig_hi68"],
                 color=_CB[0], alpha=0.45, label="68%")
  a.plot(ell, cmb["sig_med"], color=_CB[3], lw=1.0, label="median")
  a.axhline(0.0, color=_CB[3], lw=0.5, ls=":")
  a.set_xlabel(r"$\ell$")
  a.set_ylabel(r"$(\hat C_\ell - C_\ell)\,/\,\sigma_\ell$")
  a.set_title(f"{spec}: residual in cosmic-variance units")
  a.legend(fontsize=8)
  # bottom left: the worst-cosmology overlay. te crosses zero, so the
  # log scale applies only to the strictly positive spectra.
  a = ax[1, 0]
  w = cmb["worst"]
  a.plot(ell, w["truth"], color=_CB[3], lw=1.2, label="truth")
  a.plot(ell, w["pred"], color=_CB[1], lw=1.0, ls="--",
         label="emulator")
  if np.all(np.asarray(w["truth"]) > 0):
    a.set_yscale("log")
  a.set_xlabel(r"$\ell$")
  a.set_ylabel(rf"$C_\ell$ ({cmb['units']})")
  a.set_title(f"worst val cosmology (chi2 = {w['dchi2']:.1f})")
  a.legend(fontsize=8)
  # bottom right: that cosmology's per-multipole residual/sigma.
  a = ax[1, 1]
  sigma_resid = np.asarray(w["pred"] - w["truth"], dtype="float64")
  # reconstruct sigma from the band statistics is lossy; the residual
  # in physical units with the median band overlaid keeps it honest.
  a.plot(ell, sigma_resid, color=_CB[0], lw=0.8)
  a.axhline(0.0, color=_CB[3], lw=0.5, ls=":")
  a.set_xlabel(r"$\ell$")
  a.set_ylabel(rf"$\hat C_\ell - C_\ell$ ({cmb['units']})")
  a.set_title("worst val cosmology: residual")
  fa.tight_layout()
  figs.append(fa)

  fb, a = plt.subplots(figsize=(9, 5))
  hp = cmb["highpass"]
  a.plot(ell, hp["median_abs_rem"], color=_CB[0], lw=1.0)
  a.set_xlabel(r"$\ell$")
  a.set_ylabel("median |high-pass remainder|  (error-bar units)")
  a.set_title(
    f"{spec}: short-period residual content (the roughness band; "
    f"periods < ~{hp['period_cut']} in ell). The acoustic structure "
    "(period ~200-300, incl. lensing peak smoothing) is filtered "
    "out — content here is network wiggle, not physics.")
  fb.tight_layout()
  figs.append(fb)
  return figs


def _scalar_pages(sc):
  """Build the scalar-family pages from scalar_output_diagnostic.

  Page A: per-output truth-vs-predicted scatter with the identity line.
  Page B: per-output residual histograms, physical units (left) and
  standardized units (right) side by side.
  Page C: residual (standardized) vs each input parameter — the bias
  hunt: any trend says the emulator is systematically wrong along that
  direction, not just noisy.

  Arguments:
    sc = the dict scalar_output_diagnostic returned.

  Returns:
    a list of matplotlib figures.
  """
  names  = list(sc["names"])
  truth  = np.asarray(sc["truth"], dtype="float64")
  pred   = np.asarray(sc["pred"], dtype="float64")
  rstd   = np.asarray(sc["resid_std"], dtype="float64")
  params = np.asarray(sc["params"], dtype="float64")
  pnames = list(sc["param_names"])
  n_out  = len(names)
  figs = []

  # page A: truth vs predicted, identity line.
  fa, axs = plt.subplots(1, n_out, figsize=(4.5 * n_out, 4.5),
                         squeeze=False)
  for j, nm in enumerate(names):
    a = axs[0, j]
    a.scatter(truth[:, j], pred[:, j], s=4, alpha=0.4, color=_CB[0])
    lo = min(truth[:, j].min(), pred[:, j].min())
    hi = max(truth[:, j].max(), pred[:, j].max())
    a.plot([lo, hi], [lo, hi], color=_CB[3], lw=0.8, ls="--")
    a.set_xlabel(f"true {nm}")
    a.set_ylabel(f"predicted {nm}")
    a.set_title(nm)
  fa.tight_layout()
  figs.append(fa)

  # page B: residual histograms, physical | standardized.
  fb, axs = plt.subplots(n_out, 2, figsize=(11, 3.6 * n_out),
                         squeeze=False)
  for j, nm in enumerate(names):
    r_phys = pred[:, j] - truth[:, j]
    a = axs[j, 0]
    a.hist(r_phys, bins=60, color=_CB[0], alpha=0.85)
    a.set_xlabel(f"{nm}: predicted - true (physical units)")
    a.set_ylabel("val points")
    a = axs[j, 1]
    a.hist(rstd[:, j], bins=60, color=_CB[1], alpha=0.85)
    a.set_xlabel(f"{nm}: residual / training scale (standardized)")
    a.set_ylabel("val points")
  fb.tight_layout()
  figs.append(fb)

  # page C: standardized residual vs each input parameter (the bias
  # hunt): rows = outputs, columns = inputs.
  n_par = len(pnames)
  fc, axs = plt.subplots(n_out, n_par,
                         figsize=(3.2 * n_par, 3.0 * n_out),
                         squeeze=False)
  for j, nm in enumerate(names):
    for k, pn in enumerate(pnames):
      a = axs[j, k]
      a.scatter(params[:, k], rstd[:, j], s=3, alpha=0.3,
                color=_CB[0])
      a.axhline(0.0, color=_CB[3], lw=0.5, ls=":")
      if j == n_out - 1:
        a.set_xlabel(pn)
      if k == 0:
        a.set_ylabel(f"{nm} resid (std)")
  fc.tight_layout()
  figs.append(fc)
  return figs


def _grid_pages(gd):
  """Build the grid-family pages from grid_residual_diagnostic.

  Page A (1x2 or 2x2): the per-redshift fractional-residual bands for
  the emulated background function, plus the worst-cosmology overlay
  (pred vs truth and its fractional residual).
  Page B (only for a "Hubble" artifact): the DERIVED-distance page —
  fractional D_A and D_L error bands at interior redshifts, computed
  through the real integration pipeline (emulator/background.py), so
  the page tests the path a likelihood actually consumes.

  Arguments:
    gd = the dict grid_residual_diagnostic returned.

  Returns:
    a list of matplotlib figures.
  """
  z = np.asarray(gd["z"], dtype="float64")
  q = str(gd["quantity"])
  figs = []

  fa, ax = plt.subplots(2, 2, figsize=(13, 9))
  a = ax[0, 0]
  a.fill_between(z, gd["frac_lo95"], gd["frac_hi95"],
                 color=_CB[4], alpha=0.35, label="95%")
  a.fill_between(z, gd["frac_lo68"], gd["frac_hi68"],
                 color=_CB[0], alpha=0.45, label="68%")
  a.plot(z, gd["frac_med"], color=_CB[3], lw=1.0, label="median")
  a.axhline(0.0, color=_CB[3], lw=0.5, ls=":")
  a.set_xlabel("z")
  a.set_ylabel(f"fractional {q} residual")
  a.set_title(f"{q}: fractional residual over the val set")
  a.legend(fontsize=8)
  ax[0, 1].axis("off")
  w = gd["worst"]
  a = ax[1, 0]
  a.plot(z, w["truth"], color=_CB[3], lw=1.2, label="truth")
  a.plot(z, w["pred"], color=_CB[1], lw=1.0, ls="--", label="emulator")
  a.set_xlabel("z")
  a.set_ylabel(f"{q} ({gd['units']})")
  a.set_title(f"worst val cosmology (chi2 = {w['dchi2']:.1f})")
  a.legend(fontsize=8)
  a = ax[1, 1]
  a.plot(z, (np.asarray(w["pred"]) - np.asarray(w["truth"]))
            / np.asarray(w["truth"]), color=_CB[0], lw=0.8)
  a.axhline(0.0, color=_CB[3], lw=0.5, ls=":")
  a.set_xlabel("z")
  a.set_ylabel(f"fractional {q} residual")
  a.set_title("worst val cosmology: residual")
  fa.tight_layout()
  figs.append(fa)

  d = gd.get("derived")
  if d is not None:
    fb, ax = plt.subplots(1, 2, figsize=(12, 5))
    for k, (tag, label) in enumerate((("da", "D_A"), ("dl", "D_L"))):
      a = ax[k]
      a.fill_between(d["z_eval"], d[tag + "_lo68"], d[tag + "_hi68"],
                     color=_CB[0], alpha=0.45, label="68%")
      a.plot(d["z_eval"], d[tag + "_med"], color=_CB[3], lw=1.0,
             label="median")
      a.axhline(0.0, color=_CB[3], lw=0.5, ls=":")
      a.set_xlabel("z")
      a.set_ylabel(f"fractional {label} error")
      a.set_title(f"derived {label} through the real pipeline")
      a.legend(fontsize=8)
    fb.tight_layout()
    figs.append(fb)
  return figs


def _grid2d_pages(g2):
  """Build the grid2d-family pages from grid2d_residual_diagnostic.

  Page A (1x2): the median |residual| over the validation set as a
  (z, k) surface, beside the worst validation cosmology's |residual|
  surface (same color scale, so "how bad is the worst" reads at a
  glance). Under a syren law the residual is ln(P_pred / P_truth) —
  the base cancels — so the color is the fractional error of the
  served spectrum; under law "none" it is the plain fractional
  residual.
  Page B (stacked): per-k residual bands (68/95 + median) at the
  first / middle / last stored redshift, with the worst cosmology's
  cut overlaid.

  Arguments:
    g2 = the dict grid2d_residual_diagnostic returned.

  Returns:
    a list of matplotlib figures.
  """
  z = np.asarray(g2["z"], dtype="float64")
  k = np.asarray(g2["k"], dtype="float64")
  q = str(g2["quantity"])
  res_label = ("ln(pred / truth)" if g2["res_kind"] == "ln-ratio"
               else "fractional residual")
  figs = []

  med_abs = np.asarray(g2["med_abs"], dtype="float64")
  worst_abs = np.abs(np.asarray(g2["worst"]["res"], dtype="float64"))
  vmax = max(float(med_abs.max()), float(worst_abs.max()), 1e-12)
  fa, ax = plt.subplots(1, 2, figsize=(13, 5))
  for a, surf, title in ((ax[0], med_abs,
                          f"{q}: median |{res_label}| over the val set"),
                         (ax[1], worst_abs,
                          "worst val cosmology (chi2 = "
                          f"{g2['worst']['dchi2']:.1f})")):
    pc = a.pcolormesh(k, z, surf, shading="auto", cmap="viridis",
                      vmin=0.0, vmax=vmax)
    a.set_xscale("log")
    a.set_xlabel("k (1/Mpc)")
    a.set_ylabel("z")
    a.set_title(title, fontsize=10)
    fa.colorbar(pc, ax=a, label=f"|{res_label}|")
  fa.tight_layout()
  figs.append(fa)

  cuts = g2["slices"]
  fb, ax = plt.subplots(len(cuts), 1, figsize=(9, 3.2 * len(cuts)),
                        sharex=True)
  if len(cuts) == 1:
    ax = [ax]
  for a, s in zip(ax, cuts):
    a.fill_between(k, s["lo95"], s["hi95"], color=_CB[4], alpha=0.35,
                   label="95%")
    a.fill_between(k, s["lo68"], s["hi68"], color=_CB[0], alpha=0.45,
                   label="68%")
    a.plot(k, s["med"], color=_CB[3], lw=1.0, label="median")
    a.plot(k, np.asarray(g2["worst"]["res"])[s["iz"]], color=_CB[1],
           lw=0.8, ls="--", label="worst cosmology")
    a.axhline(0.0, color=_CB[3], lw=0.5, ls=":")
    a.set_xscale("log")
    a.set_ylabel(res_label)
    a.set_title(f"z = {s['z']:.3g}", fontsize=9)
    a.legend(fontsize=7)
  ax[-1].set_xlabel("k (1/Mpc)")
  fb.tight_layout()
  figs.append(fb)
  return figs


def plot_diagnostics(train_losses,
                     medians,
                     means,
                     fracs,
                     thresholds,
                     coverage,
                     floor=None,
                     hard_dir=None,
                     val_set=None,
                     names=None,
                     cuts=None,
                     cmb=None,
                     scalar=None,
                     grid=None,
                     grid2d=None,
                     savepath=None):
  """
  All available diagnostics as a single multipage figure / PDF.

  Page 1 (2x2): the training history (loss curves; fraction over
    each delta-chi2 threshold vs epoch) and the coverage diagnostic
    (hardness vs local sparsity; good/bad sparsity histograms).
  Page 2: the local-linear data-only floor (model vs floor
    delta-chi2), if `floor` is given.
  Page 3: the hard-direction regression (univariate ranking and
    joint log-linear coefficients), if `hard_dir` is given.
  Page 4: the getdist lcdm triangle of the val cosmologies, colored
    by log10 delta-chi2, if `val_set` and `names` are given.
  Page 5: the first two ln-parameter PCA directions of the val
    cosmologies, colored the same way (a principal direction in ln
    space is a product of parameter powers), same condition.
  Page 6: the same PCA plane colored by local training sparsity
    (coverage's knn_dist), with the fitted sparsity direction
    annotated, names the combinations where training is thin,
    independent of the chi2; same condition.

  Family pages, appended after the shared pages when their dict is
  given:
    cmb    -> two CMB pages (per-multipole residual bands + the worst
              overlay; the high-pass wiggle content).
    scalar -> three scalar pages (truth-vs-predicted; residual
              histograms physical + standardized; residual vs each
              input parameter).
    grid   -> the background pages (per-redshift residual bands +
              worst overlay; for a Hubble artifact the derived
              D_A / D_L page through the real pipeline).
    grid2d -> two matter-power pages (the (z, k) |residual| surfaces,
              median + worst; per-k bands at three redshifts).
  A run passes only its own family's dict; the others stay at their
  None defaults and add no pages, so a cosmic-shear run's PDF carries
  no family pages at all.

  floor / hard_dir / val_set are optional so a run can drop a page
  it cannot produce (e.g. the local-linear floor is defined only for
  a plain chi2fn, so a --rescale run omits it).

  Arguments:
    train_losses, medians, means, fracs, thresholds = the
      run_emulator histories (see plot_history).
    coverage = the dict coverage_diagnostic returned.
    floor    = the dict local_linear_floor returned, or None.
    hard_dir = the dict hard_direction_regression returned, or None.
    val_set  = the validation source dict ("C" / "idx"), or None;
               its rows must be the ones coverage's dchi2 scored.
    names    = parameter column names in the dump's order, or None.
    cuts     = optional physical-cut values ("omegabh2_hi" /
               "omegam2h2_lo" / "omegam2h2_hi") to shade gray on the
               triangle page (empty cut regions then read as removed,
               not as failures).
    savepath = if given, write a (multipage) PDF there and close;
               if None, show each page interactively.
  """
  figs = []
  # page 1: history (top row) + coverage (bottom row).
  f1, ax = plt.subplots(2, 2, figsize=(12, 9))
  _history_panels(ax[0, 0], ax[0, 1], train_losses, medians,
                  means, fracs, thresholds)
  _coverage_panels(ax[1, 0], ax[1, 1], coverage["knn_dist"],
                   coverage["dchi2"], coverage["k_nn"])
  f1.tight_layout()
  figs.append(f1)

  # page 2: the local-linear data floor (plain chi2fn only).
  if floor is not None:
    f2, a2 = plt.subplots(figsize=(6, 6))
    _floor_panel(a2, floor)
    f2.tight_layout()
    figs.append(f2)

  # page 3: the hard-direction regression.
  if hard_dir is not None:
    f3, a3 = plt.subplots(1, 2, figsize=(13, 6))
    _hard_direction_panels(a3[0], a3[1], hard_dir)
    f3.tight_layout()
    figs.append(f3)

  # pages 4-6: where in parameter space the failures live. Page 4 is
  # the lcdm triangle (getdist lays it out itself, so no
  # tight_layout); page 5 the ln-parameter PCA plane colored by chi2
  # (hardness); page 6 the same plane colored by local training
  # sparsity (coverage), with the fitted sparsity direction,
  # aligned gradients on 5 and 6 say the failures are coverage,
  # diverging ones say the hardness is intrinsic.
  if val_set is not None and names is not None:
    f4 = _lcdm_triangle_fig(source=val_set,
                            names=names,
                            dchi2=coverage["dchi2"],
                            cuts=cuts)
    if f4 is not None:
      figs.append(f4)
    f5 = _lnparam_pca_fig(source=val_set,
                          names=names,
                          color=_log_dchi2_color(coverage["dchi2"]),
                          clabel=r"$\log_{10}\Delta\chi^2$",
                          title="ln-parameter PCA of the val "
                                "cosmologies, colored by hardness",
                          fit_target=np.maximum(
                            np.asarray(coverage["dchi2"],
                                       dtype="float64"), 1e-12),
                          vmin=_CHI2_CBAND[0],
                          vmax=_CHI2_CBAND[1])
    if f5 is not None:
      figs.append(f5)
    # sparsity color: percentile-clipped so one far outlier does not
    # own the colorbar; the direction fit uses the unclipped values.
    knn = np.asarray(coverage["knn_dist"], dtype="float64")
    lo, hi = np.percentile(knn, [1.0, 99.0])
    f6 = _lnparam_pca_fig(source=val_set,
                          names=names,
                          color=np.clip(knn, lo, hi),
                          clabel=(f"mean dist to {coverage['k_nn']} "
                                  "nearest train pts (whitened)"),
                          title="ln-parameter PCA, colored by "
                                "training sparsity",
                          fit_target=knn,
                          vmin=lo,
                          vmax=hi)
    if f6 is not None:
      figs.append(f6)

  # family pages: appended after the shared pages;
  # None (the default) adds nothing, keeping the cosmic-shear PDF
  # byte-identical.
  if cmb is not None:
    figs.extend(_cmb_pages(cmb))
  if scalar is not None:
    figs.extend(_scalar_pages(scalar))
  if grid is not None:
    figs.extend(_grid_pages(grid))
  if grid2d is not None:
    figs.extend(_grid2d_pages(grid2d))

  _save_pages(figs, savepath)


def source_param_samples(source, names, labels, label):
  """
  getdist MCSamples of one source's cosmological parameters.

  Pulls the rows the source actually uses (source["idx"]) from its
  parameter dump and wraps them as equally-weighted samples for a
  coverage triangle (no likelihood, no chi2). Reads no module
  globals, source, names, labels, and the legend label all arrive
  as arguments.

  Arguments:
    source = source dict with "C" (full param dump) and "idx"
             (global rows actually in use).
    names  = parameter column names, in the dump's column
             order (pgeom.names).
    labels = LaTeX labels for those columns (no surrounding $).
    label  = legend label for this set (e.g. "train").

  Returns:
    an MCSamples over the source's used parameter rows.
  """
  # the rows this source uses, coverage is about what was
  # trained / validated on, not the whole file.
  rows = np.sort(source["idx"])
  # raw physical parameters of those rows (never whitened).
  P = np.asarray(source["C"][rows], dtype="float64")
  return MCSamples(samples=P, 
                   names=names, 
                   labels=labels,
                   label=label,
                   settings={"smooth_scale_1D": 0.3,
                             "smooth_scale_2D": 0.3,
                             "fine_bins_2D": 512})


def dv_to_xi(dv_row, geom):
  """
  Reshape one full data-vector row into the (theta, xip, xim)
  matrix layout of plot_xi, using its cosmic-shear block.

  Takes the leading xi_size entries (xi_plus then xi_minus, pairs
  (i<=j) outer / theta inner) and scatters each pair's ntheta values
  into the (i, j) slot of an (ntheta, ntomo, ntomo) array (upper
  triangle filled; the rest stay 0 and plot_xi never reads them).

  Arguments:
    dv_row = (total_size,) full data vector; only the leading
             geom.xi_size cosmic-shear entries are used.
    geom   = geometry carrying ntheta / source_ntomo /
             theta_centers / xi_size.
  Returns:
    (theta, xip, xim): theta (ntheta,) [arcmin]; xip, xim
    (ntheta, ntomo, ntomo).
  """
  nt    = geom.source_ntomo
  ntha  = geom.ntheta
  block = np.asarray(dv_row[:geom.xi_size], dtype="float64")
  pairs = []
  for i in range(nt):
    for j in range(i, nt):
      pairs.append((i, j))
  half  = len(pairs) * ntha
  xip = np.zeros((ntha, nt, nt))
  xim = np.zeros((ntha, nt, nt))
  for p, (i, j) in enumerate(pairs):
    xip[:, i, j] = block[p * ntha:(p + 1) * ntha]
    xim[:, i, j] = block[half + p * ntha:
                         half + (p + 1) * ntha]
  return (geom.theta_centers, xip, xim)


def plot_xi(pm, xi, xi_ref = None, param = None, colorbarlabel = None,
            marker = None, linestyle = None, linewidth = None,
            ylim = [0.88,1.12], cmap = "viridis", legend = None,
            legendloc = (0.6,0.78), yaxislabelsize = 16, yaxisticklabelsize = 10,
            xaxisticklabelsize = 20, bintextpos = [[0.8, 0.875],[0.2,0.875]],
            bintextsize = 15, figsize = (12, 12), show = None, thetashow=[3,1000],
            colorbar=1):
    """
    Tomographic grid of xi+ / xi- curves (a visual check, not a
    diagnostic page). One panel per redshift-bin pair: xi+ on the
    lower triangle, xi- on the upper; absolute curves when xi_ref
    is None, ratios xi / xi_ref otherwise. Ported byte-faithfully
    from the notebook, so its body keeps the original style.

    Arguments:
      pm            = integer selecting the correlation sign: pm > 0
                      draws xi+, pm <= 0 draws xi- (the body tests
                      `pm > 0`, not a string mode).
      xi            = list of (theta, xip, xim) triples (dv_to_xi
                      output), one curve set per line drawn.
      xi_ref        = reference triple; when given, panels show the
                      ratio xi / xi_ref and share both axes.
      param         = per-curve scalar (e.g. a cosmology parameter)
                      that colors the curves through cmap; length
                      must match xi.
      colorbarlabel = label for the param colorbar.
      marker        = marker cycle (list) or None.
      linestyle     = linestyle cycle (list); solid when None.
      linewidth     = linewidth cycle (list) or None.
      ylim          = y range of the ratio panels.
      cmap          = matplotlib colormap name for param coloring;
                      the default "viridis" is sequential and
                      colorblind-safe (the palette rule).
      legend        = per-curve legend labels or None.
      legendloc     = legend anchor (axes fraction).
      yaxislabelsize / yaxisticklabelsize / xaxisticklabelsize
                    = font sizes for the y label and tick labels.
      bintextpos    = [[x, y] lower, [x, y] upper] axes-fraction
                      anchors of the per-panel bin annotation.
      bintextsize   = font size of that annotation.
      figsize       = figure size in inches.
      show          = display toggle: not None calls fig.show() and
                      returns None; None returns (fig, axes).
      thetashow     = [min, max] theta range (arcmin) shown.
      colorbar      = 1 to draw the param colorbar, None to skip.

    Returns:
      (fig, axes) when show is None; None when show is set (the
      figure is displayed instead); 0 (int) on malformed input,
      after printing a message.
    """

    (theta, xip, xim) = xi[0]
    (ntheta, ntomo, ntomo2) = xip.shape    

    if ntomo != ntomo2:
        print("Bad Input (ntomo)")
        return 0
            
    if ntheta != len(theta):
        print("Bad Input (theta)")
        return 0

    if xi_ref is None:
        fig, axes = plt.subplots(
            nrows = ntomo, 
            ncols = ntomo, 
            figsize = figsize, 
            sharex = True, 
            sharey = False, 
            gridspec_kw = {'wspace': 0.25, 'hspace': 0.05})
    else:
        fig, axes = plt.subplots(
            nrows = ntomo, 
            ncols = ntomo, 
            figsize = figsize, 
            sharex = True, 
            sharey = True, 
            gridspec_kw = {'wspace': 0.0, 'hspace': 0.0})    

    cm = plt.get_cmap(cmap)

    if not (param is None or colorbar is None):
        norm = matplotlib.colors.Normalize(vmin=param[0],
                                           vmax=param[-1])
        cb = fig.colorbar(
            matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax = axes.ravel().tolist(), 
            orientation = 'vertical', 
            aspect = 50, 
            pad = -0.16, 
            shrink = 0.5
        )
        if not (colorbarlabel is None):
            cb.set_label(label = colorbarlabel, 
                         size = 20, 
                         weight = 'bold', 
                         labelpad = 2)
        if len(param) != len(xi):
            print("Bad Input")
            return 0

    if not (marker is None):
        markercycler = itertools.cycle(marker)
    
    if not (linestyle is None):
        linestylecycler = itertools.cycle(linestyle)
    else:
        linestylecycler = itertools.cycle(['solid'])
    
    if not (linewidth is None):
        linewidthcycler = itertools.cycle(linewidth)
    else:
        linewidthcycler = itertools.cycle([1.0])
        
    for i in range(ntomo):
        for j in range(ntomo):
            if i>j:                
                axes[j,i].axis('off')
            else:
                ximin = []
                ximax = []
                for (theta, xip, xim) in xi:
                    if pm > 0:
                        ximin.append(np.min(theta*xip[:,i,j]*10**4))
                        ximax.append(np.max(theta*xip[:,i,j]*10**4))
                    else:
                        ximin.append(np.min(theta*xim[:,i,j]*10**4))
                        ximax.append(np.max(theta*xim[:,i,j]*10**4))
                        
                axes[j,i].set_xlim(thetashow)
                
                if xi_ref is None:
                    axes[j,i].set_ylim([np.min(ylim[0]*np.array(ximin)), 
                                        np.max(ylim[1]*np.array(ximax))])
                else:
                    tmp = np.array(ylim) - 1
                    axes[j,i].set_ylim(tmp.tolist())
                axes[j,i].set_xscale('log')
                axes[j,i].set_yscale('linear')
                
                if i == 0:
                    if xi_ref is None:
                        if pm > 0:
                            axes[j,i].set_ylabel(r"$\theta \xi_{+} \times 10^4$", 
                                                 fontsize=yaxislabelsize)
                        else:
                            axes[j,i].set_ylabel(r"$\theta \xi_{-} \times 10^4$", 
                                                 fontsize=yaxislabelsize)
                    else:
                        if pm > 0:
                            axes[j,i].set_ylabel(r"frac. diff. ($\xi_{+})$", 
                                                 fontsize=yaxislabelsize)
                        else:
                            axes[j,i].set_ylabel(r"frac. diff. ($\xi_{-})$", 
                                                 fontsize=yaxislabelsize)

                if j == ntomo-1:
                    axes[j,i].set_xlabel(r"$\theta$ [arcmin]", fontsize=16)
                for item in (axes[j,i].get_yticklabels()):
                    item.set_fontsize(yaxisticklabelsize)
                for item in (axes[j,i].get_xticklabels()):
                    item.set_fontsize(xaxisticklabelsize)

                if pm > 0:
                    axes[j,i].text(bintextpos[0][0], 
                                   bintextpos[0][1], 
                                   "$(" +  str(i) + "," +  str(j) + ")$", 
                                   horizontalalignment='center', 
                                   verticalalignment='center',
                                   fontsize=bintextsize,
                                   usetex=True,
                                   transform=axes[j,i].transAxes)
                else:
                    axes[j,i].text(bintextpos[1][0], 
                                   bintextpos[1][1], 
                                   "$(" +  str(i) + "," +  str(j) + ")$", 
                                   horizontalalignment='center', 
                                   verticalalignment='center',
                                   fontsize=15,
                                   usetex=True,
                                   transform=axes[j,i].transAxes)

                if xi_ref is None:
                    # plot(x, y, ...): x = theta, y = theta *
                    # xi_+/- * 1e4 (the scaled correlation fn).
                    for x, (theta, xip, xim) in enumerate(xi):
                        if pm > 0:
                            if marker is None:
                                axes[j,i].plot(theta, 
                                               theta*xip[:,i,j]*10**4, 
                                               color=cm(x/len(xi)), 
                                               linewidth=next(linewidthcycler), 
                                               linestyle=next(linestylecycler))
                            else:
                                axes[j,i].plot(theta, 
                                               theta*xip[:,i,j]*10**4, 
                                               color=cm(x/len(xi)), 
                                               markerfacecolor='None', 
                                               marker=next(markercycler), 
                                               markeredgecolor=cm(x/len(xi)), 
                                               linestyle='None', 
                                               markersize=3)
                        else:
                            if marker is None:   
                                axes[j,i].plot(theta, theta*xim[:,i,j]*10**4, 
                                               color=cm(x/len(xi)), 
                                               linewidth=next(linewidthcycler), 
                                               linestyle=next(linestylecycler))
                            else:
                                axes[j,i].plot(theta, 
                                               theta*xim[:,i,j]*10**4, 
                                               color=cm(x/len(xi)), 
                                               markerfacecolor='None', 
                                               marker=next(markercycler), 
                                               markeredgecolor=cm(x/len(xi)), 
                                               linestyle='None', 
                                               markersize=3)
                else:
                    (theta_ref, xip_ref, xim_ref) = xi_ref
                    # plot(x, y, ...): x = theta, y = xi_+/- /
                    # xi_ref - 1 (the fractional difference).
                    for x, (theta, xip, xim) in enumerate(xi):
                        if not np.array_equal(theta, theta_ref):
                            print("inconsistent theta bins")
                            return 0
                        if pm > 0:
                            if marker is None:
                                axes[j,i].plot(theta, xip[:,i,j]/xip_ref[:,i,j]-1.0, 
                                               color=cm(x/len(xi)), 
                                               linewidth=next(linewidthcycler), 
                                               linestyle=next(linestylecycler))
                            else:
                                axes[j,i].plot(theta, 
                                               xip[:,i,j]/xip_ref[:,i,j]-1.0, 
                                               color=cm(x/len(xi)), 
                                               markerfacecolor='None',
                                               marker=next(markercycler),  
                                               markeredgecolor=cm(x/len(xi)), 
                                               linestyle='None', 
                                               markersize=3)
                        else:
                            if marker is None:   
                                lines = axes[j,i].plot(theta, 
                                                       xim[:,i,j]/xim_ref[:,i,j]-1.0, 
                                                       color=cm(x/len(xi)), 
                                                       linewidth=next(linewidthcycler), 
                                                       linestyle=next(linestylecycler))
                            else:
                                axes[j,i].plot(theta, 
                                               xim[:,i,j]/xim_ref[:,i,j]-1.0, 
                                               color=cm(x/len(xi)), 
                                               markerfacecolor='None', 
                                               marker=next(markercycler), 
                                               markeredgecolor=cm(x/len(xi)), 
                                               linestyle='None', markersize=3)    
    if not (legend is None):
        if len(legend) != len(xi):
            print("Bad Input")
            return 0
        fig.legend(legend, 
                   loc=legendloc,
                   borderpad=0.1,
                   handletextpad=0.4,
                   handlelength=1.5,
                   columnspacing=0.35,
                   scatteryoffsets=[0],
                   frameon=False)  
    if not (show is None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig.show()
    else:
        return (fig, axes)
