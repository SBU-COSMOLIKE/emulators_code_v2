#!/usr/bin/env python3
"""triangle-shading (optional): each physical cut shades its own panel.

WHAT: the grey cut-window shading on the diagnostics corner plot. WHY:
the shading tells a reader which parameter regions the density cuts
removed. A fill on the wrong panel would attach that warning to the
wrong pair of cosmological parameters. HOW: render a synthetic-sample
triangle headlessly through the real plotting helper, identify every
Matplotlib Axes by its x and y parameters, and compare the observed
``(x parameter, y parameter, window)`` set with an independently
written table of the physical formulas. The check also verifies the
colour of each individual artist and the two bands on the
``omegamh2`` diagonal marginal.

The mutation leg physically moves one correct filled artist from its
Axes to a wrong Axes. The total number of filled artists, shaded panels,
grey artists, and marginal bands stays fixed. The exact panel/window
check must still reject the altered figure.
"""

import sys
from unittest import mock

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from emulator import plotting


FAILURES = []


# GetDist orders the synthetic fixture by the alias table in plotting.py.
# These names are scientific coordinates, not rendered LaTeX strings. The
# gate uses them to give every Matplotlib Axes a stable identity.
PLOT_PARAMETERS = ("ns", "h0", "omegab", "omegam", "omegamh2")


# This table is the gate's independent scientific reference. The outer tuple
# is an immutable ordered sequence. Each inner three-item tuple says that one
# physical window is a function of the named x and y coordinates and therefore
# must be drawn on that two-dimensional panel. It does not call or copy a
# result from plotting._window_masks.
EXPECTED_PANEL_WINDOWS = (
  ("h0", "omegab", "omegabh2"),
  ("h0", "omegam", "omegam2h2"),
  ("h0", "omegam", "omegamh2"),
  ("ns", "omegamh2", "omegamh2"),
  ("ns", "omegamh2", "omegamh2ns"),
  ("h0", "omegamh2", "omegam2h2"),
  ("h0", "omegamh2", "omegamh2"),
  ("omegab", "omegamh2", "omegamh2"),
  ("omegam", "omegamh2", "omegam2h2"),
  ("omegam", "omegamh2", "omegamh2"),
)


def report(label, ok, detail, aid=None):
  """Print one acceptance line and record a failure."""
  if ok:
    mark = "PASS"
  else:
    mark = "FAIL"
  print("  [" + mark + "] " + label + "  (" + detail + ")")
  # (queue 2) the per-leg assertion manifest the board folds into this gate's
  # executed set: one reserved '##AID <aid> <result>' line per acceptance leg.
  # The child's exit status stays the single aggregate verdict, not a leg.
  if aid is not None:
    print("##AID " + aid + " " + mark)
  if not ok:
    FAILURES.append(label)


def synthetic_source(n_rows):
  """Build a synthetic LCDM parameter scatter that crosses every cut.

  Arguments:
    n_rows = number of synthetic cosmologies.

  Returns:
    (source, names, dchi2): source = {"C", "idx"} with columns
    omegab / omegam / h0 / ns in physical ranges that straddle the cut
    bounds; names gives the column names; dchi2 is a positive colour array.
  """
  rng = np.random.default_rng(0)
  omegab = rng.normal(
    0.048,
    0.004,
    n_rows)
  omegam = rng.normal(
    0.30,
    0.05,
    n_rows)
  h0 = rng.normal(
    67.0,
    5.0,
    n_rows)
  ns = rng.normal(
    0.96,
    0.02,
    n_rows)
  columns = np.column_stack([omegab, omegam, h0, ns]).astype("float64")
  source = {"C": columns, "idx": np.arange(n_rows)}
  names = ["omegab", "omegam", "h0", "ns"]
  dchi2 = rng.uniform(
    0.01,
    1.0,
    n_rows)
  return source, names, dchi2


def cut_config():
  """Return the four active physical windows used by this fixture."""
  return {
    "omegabh2_lo": 0.014,
    "omegabh2_hi": 0.035,
    "omegam2h2_lo": 0.015,
    "omegam2h2_hi": 0.08,
    "omegamh2_lo": 0.10,
    "omegamh2_hi": 0.18,
    "omegamh2ns_lo": 0.10,
    "omegamh2ns_hi": 0.17,
  }


def close_to_grey(rgba):
  """Return True when one RGBA colour matches plotting._CUT_GREY."""
  grey = np.asarray(plotting._CUT_GREY)
  return bool(np.allclose(
    np.asarray(rgba),
    grey,
    atol=1.0e-6))


def render_with_contour_trace(source, names, dchi2, cuts):
  """Render the real triangle and retain each cut-mask drawing call.

  ``Axes.contourf`` is the Matplotlib method that turns a two-dimensional
  boolean mask into a filled artist. ``mock.patch.object`` temporarily
  replaces that method with the wrapper below. The wrapper first calls the
  real method, then stores the mask and returned artist when the artist uses
  the cut layer's z-order zero. Leaving the ``with`` block restores the real
  method even if rendering raises an exception.

  The trace is observation only. The production code receives the same
  arrays and the same return value as an unpatched call.

  Returns:
    (figure, trace): figure is the Matplotlib Figure. trace is a list of
    dictionaries. Each dictionary owns one artist, its x/y mesh grids and
    the boolean exclusion mask that produced it.
  """
  trace = []
  real_contourf = Axes.contourf

  def traced_contourf(ax, *args, **kwargs):
    artist = real_contourf(
      ax,
      *args,
      **kwargs)
    if kwargs.get("zorder") == 0:
      if len(args) < 3:
        raise RuntimeError(
          "the cut-layer contourf call must provide x, y, and mask arrays")
      trace.append({
        "artist": artist,
        "x_grid": np.asarray(args[0]).copy(),
        "y_grid": np.asarray(args[1]).copy(),
        "mask": np.asarray(args[2]) > 0.5,
      })
    return artist

  with mock.patch.object(
      Axes,
      "contourf",
      new=traced_contourf):
    figure = plotting._lcdm_triangle_fig(
      source=source,
      names=names,
      dchi2=dchi2,
      cuts=cuts)
  return figure, trace


def ordered_unique_positions(values, reverse=False):
  """Group numerically equal subplot coordinates and sort the groups.

  ``sorted`` builds a new eager list. It does not change the input sequence.
  The nested loops then compare each new coordinate with every group already
  retained.
  """
  unique = []
  for value in sorted(
      values,
      reverse=reverse):
    already_present = False
    for old_value in unique:
      if np.isclose(
          value,
          old_value,
          rtol=0.0,
          atol=1.0e-10):
        already_present = True
        break
    if not already_present:
      unique.append(value)
  return unique


def position_index(value, positions):
  """Return which subplot-grid position contains one coordinate."""
  distances = []
  for position in positions:
    distances.append(abs(value - position))
  index = int(np.argmin(distances))
  if distances[index] > 1.0e-10:
    raise ValueError("an Axes does not sit on the inferred triangle grid")
  return index


def triangle_axis_identities(figure, plot_parameters):
  """Map each triangle Axes object to its x and y parameter names.

  Matplotlib calls the whole page a ``Figure``. Each panel inside that page
  is an ``Axes`` object. GetDist also adds a narrow colour-bar Axes, so the
  gate cannot treat every Figure axis as a parameter panel.

  All triangle panels have the same width and height. For ``n`` plotted
  parameters there are ``n(n+1)/2`` such panels. Their left edges identify
  x coordinates from left to right. Their bottom edges identify y
  coordinates from top to bottom. The returned dictionary uses the Axes
  objects themselves as keys, so later checks follow an artist's actual
  owner after a mutation moves it.
  """
  n_parameters = len(plot_parameters)
  expected_panel_count = n_parameters * (n_parameters + 1) // 2

  panel_axes = None
  for candidate in figure.axes:
    candidate_box = candidate.get_position()
    same_size = []
    for axis in figure.axes:
      axis_box = axis.get_position()
      same_width = np.isclose(
        axis_box.width,
        candidate_box.width,
        rtol=0.0,
        atol=1.0e-10)
      same_height = np.isclose(
        axis_box.height,
        candidate_box.height,
        rtol=0.0,
        atol=1.0e-10)
      if same_width and same_height:
        same_size.append(axis)
    if len(same_size) == expected_panel_count:
      panel_axes = same_size
      break

  if panel_axes is None:
    raise ValueError(
      "the Figure does not contain one complete equal-size triangle grid")

  panel_left_edges = []
  panel_bottom_edges = []
  for axis in panel_axes:
    axis_box = axis.get_position()
    panel_left_edges.append(axis_box.x0)
    panel_bottom_edges.append(axis_box.y0)
  left_edges = ordered_unique_positions(
    panel_left_edges)
  bottom_edges = ordered_unique_positions(
    panel_bottom_edges,
    reverse=True)
  if len(left_edges) != n_parameters or len(bottom_edges) != n_parameters:
    raise ValueError(
      "the triangle grid does not have one row and column per parameter")

  identities = {}
  for axis in panel_axes:
    box = axis.get_position()
    x_index = position_index(
      box.x0,
      left_edges)
    y_index = position_index(
      box.y0,
      bottom_edges)
    identities[axis] = (plot_parameters[x_index], plot_parameters[y_index])

  expected_identities = set()
  for y_index in range(len(plot_parameters)):
    y_parameter = plot_parameters[y_index]
    for x_index in range(y_index + 1):
      expected_identities.add((plot_parameters[x_index], y_parameter))
  if set(identities.values()) != expected_identities:
    raise ValueError(
      "the inferred Axes identities do not form the complete lower triangle")
  return identities


def expected_quantity(x_parameter, y_parameter, window, x_grid, y_grid):
  """Compute one window quantity from the named panel coordinates.

  This function is deliberately explicit. Each branch is one equation from
  the coverage table. It does not ask the plotting module which window it
  intended to draw.
  """
  identity = (x_parameter, y_parameter, window)
  if identity == ("h0", "omegab", "omegabh2"):
    return y_grid * (x_grid / 100.0) ** 2

  if identity == ("h0", "omegam", "omegam2h2"):
    return (y_grid * x_grid / 100.0) ** 2
  if identity == ("h0", "omegam", "omegamh2"):
    return y_grid * (x_grid / 100.0) ** 2

  if identity == ("ns", "omegamh2", "omegamh2"):
    return y_grid
  if identity == ("ns", "omegamh2", "omegamh2ns"):
    return y_grid * x_grid

  if identity == ("h0", "omegamh2", "omegam2h2"):
    return y_grid ** 2 / (x_grid / 100.0) ** 2
  if identity == ("h0", "omegamh2", "omegamh2"):
    return y_grid

  if identity == ("omegab", "omegamh2", "omegamh2"):
    return y_grid

  if identity == ("omegam", "omegamh2", "omegam2h2"):
    return x_grid * y_grid
  if identity == ("omegam", "omegamh2", "omegamh2"):
    return y_grid

  raise ValueError("no independent formula exists for " + repr(identity))


def expected_exclusion_mask(identity, x_grid, y_grid, cuts):
  """Evaluate one strict cut window on a panel mesh."""
  x_parameter, y_parameter, window = identity
  quantity = expected_quantity(
    x_parameter=x_parameter,
    y_parameter=y_parameter,
    window=window,
    x_grid=x_grid,
    y_grid=y_grid)
  lower = cuts[window + "_lo"]
  upper = cuts[window + "_hi"]
  return (quantity <= lower) | (quantity >= upper)


def panel_shading_errors(trace, axis_identities, cuts):
  """Return every mismatch between observed and expected panel shading."""
  errors = []
  observed = []
  expected_set = set(EXPECTED_PANEL_WINDOWS)

  # A complete census compares the traced contourf objects with every
  # z-order-zero collection currently attached to the triangle axes. This
  # detects both an untraced extra fill and a traced artist that disappeared.
  # id(object) is Python's identity number for that exact in-memory object.
  # Comparing identity numbers avoids asking Matplotlib artists to define an
  # element-by-element equality operation.
  traced_ids = set()
  for call in trace:
    traced_ids.add(id(call["artist"]))
  attached_ids = set()
  for axis in axis_identities:
    for collection in axis.collections:
      if collection.get_zorder() == 0:
        attached_ids.add(id(collection))
  if traced_ids != attached_ids:
    errors.append("the contour trace and attached cut-layer artists differ")

  for call_index in range(len(trace)):
    call = trace[call_index]
    artist = call["artist"]
    axis = artist.axes
    if axis not in axis_identities:
      errors.append(
        "cut artist " + str(call_index) + " is not on a triangle Axes")
      continue

    x_parameter, y_parameter = axis_identities[axis]
    matching_windows = []
    for identity in EXPECTED_PANEL_WINDOWS:
      expected_x_parameter = identity[0]
      expected_y_parameter = identity[1]
      expected_window = identity[2]
      if (expected_x_parameter != x_parameter
          or expected_y_parameter != y_parameter):
        continue
      expected_mask = expected_exclusion_mask(
        identity=identity,
        x_grid=call["x_grid"],
        y_grid=call["y_grid"],
        cuts=cuts)
      if np.array_equal(
          call["mask"],
          expected_mask):
        matching_windows.append(expected_window)

    if len(matching_windows) != 1:
      errors.append(
        "artist " + str(call_index) + " on (" + x_parameter + ", "
        + y_parameter + ") matches " + str(len(matching_windows))
        + " independent window masks")
      continue
    observed.append((x_parameter, y_parameter, matching_windows[0]))

  observed_set = set(observed)
  # A set stores each tuple once and ignores order. Equality therefore asks
  # whether both sides contain exactly the same panel/window identities.
  if observed_set != expected_set:
    missing = sorted(expected_set - observed_set)
    extra = sorted(observed_set - expected_set)
    errors.append("panel/window set mismatch; missing=" + repr(missing)
                  + " extra=" + repr(extra))
  if len(observed) != len(EXPECTED_PANEL_WINDOWS):
    errors.append("expected " + str(len(EXPECTED_PANEL_WINDOWS))
                  + " distinct cut artists, observed " + str(len(observed)))
  return errors


def shading_colour_errors(axis_identities):
  """Return every cut-layer artist whose face colour is not _CUT_GREY."""
  errors = []
  for axis, identity in axis_identities.items():
    artists = []
    for collection in axis.collections:
      if collection.get_zorder() == 0:
        artists.append(collection)
    for patch in axis.patches:
      if patch.get_zorder() == 0:
        artists.append(patch)
    for artist_index in range(len(artists)):
      artist = artists[artist_index]
      facecolours = np.asarray(artist.get_facecolor())
      if facecolours.ndim == 1:
        # ``None`` inserts a length-one row axis. NumPy returns a view, so this
        # changes only how the gate indexes the values; it does not copy or
        # alter Matplotlib's RGBA data.
        facecolours = facecolours[None, :]
      if facecolours.shape[0] == 0:
        errors.append(repr(identity) + " artist " + str(artist_index)
                      + " has no face colour")
        continue
      for rgba in facecolours:
        if not close_to_grey(rgba):
          errors.append(repr(identity) + " artist " + str(artist_index)
                        + " has rgba " + repr(tuple(rgba)))
  return errors


def marginal_band_errors(axis_identities, cuts):
  """Verify both excluded intervals on the omegamh2 diagonal Axes."""
  target_identity = ("omegamh2", "omegamh2")
  target_axis = None
  patches = []
  for axis, identity in axis_identities.items():
    for patch in axis.patches:
      if patch.get_zorder() == 0:
        patches.append((identity, patch))
    if identity == target_identity:
      target_axis = axis

  errors = []
  if target_axis is None:
    return ["the omegamh2 diagonal Axes is absent"]
  wrong_axes = []
  for identity, _patch in patches:
    if identity != target_identity:
      wrong_axes.append(identity)
  if wrong_axes:
    errors.append("marginal bands appear on wrong Axes " + repr(wrong_axes))

  target_patches = []
  for identity, patch in patches:
    if identity == target_identity:
      target_patches.append(patch)
  x_min, x_max = target_axis.get_xlim()
  expected_intervals = [
    (x_min, min(cuts["omegamh2_lo"], x_max)),
    (max(cuts["omegamh2_hi"], x_min), x_max),
  ]
  observed_intervals = []
  for patch in target_patches:
    if not hasattr(patch, "get_x") or not hasattr(patch, "get_width"):
      errors.append("an omegamh2 marginal artist is not an interval patch")
      continue
    observed_intervals.append((patch.get_x(),
                               patch.get_x() + patch.get_width()))

  observed_intervals.sort()
  expected_intervals.sort()
  if len(observed_intervals) != len(expected_intervals):
    errors.append("expected two omegamh2 bands, observed "
                  + str(len(observed_intervals)))
  elif not np.allclose(
      np.asarray(observed_intervals),
      np.asarray(expected_intervals),
      rtol=0.0,
      atol=1.0e-10):
    errors.append("omegamh2 band intervals differ from the cut bounds")
  return errors


def old_global_summary(axis_identities):
  """Return the three global counts used by the former weak check."""
  artist_count = 0
  shaded_panel_count = 0
  grey_count = 0
  for axis in axis_identities:
    axis_artists = []
    for collection in axis.collections:
      if collection.get_zorder() == 0:
        axis_artists.append(collection)
    for patch in axis.patches:
      if patch.get_zorder() == 0:
        axis_artists.append(patch)
    artist_count += len(axis_artists)
    if axis_artists:
      shaded_panel_count += 1
    for artist in axis_artists:
      facecolours = np.asarray(artist.get_facecolor())
      if facecolours.ndim == 1:
        # Give one RGBA tuple the same (number of colours, 4) shape used for a
        # multi-colour artist. This is a NumPy view, not an in-place change.
        facecolours = facecolours[None, :]
      every_face_is_grey = True
      for rgba in facecolours:
        if not close_to_grey(rgba):
          every_face_is_grey = False
          break
      if every_face_is_grey:
        grey_count += 1
  return artist_count, shaded_panel_count, grey_count


def moved_artist_mutation_is_rejected(trace, axis_identities, cuts):
  """Move one real collection while preserving every old global count."""
  source_axis = None
  destination_axis = None
  for axis, identity in axis_identities.items():
    if identity == ("h0", "omegab"):
      source_axis = axis
    if identity == ("ns", "h0"):
      destination_axis = axis
  if source_axis is None or destination_axis is None:
    return False, "the source or destination Axes for the mutation is absent"

  source_artists = []
  for call in trace:
    if call["artist"].axes is source_axis:
      source_artists.append(call["artist"])
  if len(source_artists) != 1:
    return False, "the source panel does not own exactly one cut artist"

  artist = source_artists[0]
  before = old_global_summary(axis_identities)
  try:
    # remove() detaches the existing Collection from its Axes. add_collection()
    # attaches that same object to a different Axes; it does not copy it. The
    # figure therefore keeps the same number and colours of artists.
    artist.remove()
    destination_axis.add_collection(artist)
    after = old_global_summary(axis_identities)
    mutation_errors = panel_shading_errors(
      trace=trace,
      axis_identities=axis_identities,
      cuts=cuts)
    rejected = (before == after) and bool(mutation_errors)
    detail = ("old counts before/after " + repr(before) + "/" + repr(after)
              + "; exact-set errors " + str(len(mutation_errors)))
    return rejected, detail
  finally:
    # Restore the Figure because later legs inspect the original valid plot.
    if artist.axes is not None:
      artist.remove()
    source_axis.add_collection(artist)


def finish():
  """Print the aggregate verdict and return the process exit status."""
  print("")
  if len(FAILURES) == 0:
    print("triangle-shading: ALL PASS")
    return 0
  print("triangle-shading: " + str(len(FAILURES)) + " FAILURE(S)")
  return 1


def main():
  """Render the four-window triangle and assert every shading owner."""
  FAILURES.clear()
  print("== triangle-shading ==")
  source, names, dchi2 = synthetic_source(n_rows=4000)
  cuts = cut_config()

  figure, trace = render_with_contour_trace(
    source=source,
    names=names,
    dchi2=dchi2,
    cuts=cuts)
  report(
    "the triangle figure was produced",
    figure is not None,
    "None means fewer than two LCDM columns were recognized",
    aid="triangle-shading.figure-produced")
  if figure is None:
    return finish()

  try:
    axis_identities = triangle_axis_identities(
      figure=figure,
      plot_parameters=PLOT_PARAMETERS)
  except ValueError as exc:
    report(
      "each physical window is shaded on its named parameter panel",
      False,
      str(exc),
      aid="triangle-shading.panel-window-set-exact")
    plt.close(figure)
    return finish()

  panel_errors = panel_shading_errors(trace=trace,
                                      axis_identities=axis_identities,
                                      cuts=cuts)
  mutation_ok, mutation_detail = moved_artist_mutation_is_rejected(
    trace=trace,
    axis_identities=axis_identities,
    cuts=cuts)
  panel_detail = (
    "observed " + str(len(trace)) + " of "
    + str(len(EXPECTED_PANEL_WINDOWS)) + " expected artists; "
    + mutation_detail)
  if panel_errors:
    panel_detail += "; " + " | ".join(panel_errors)
  report(
    "each physical window is shaded on its named parameter panel",
    len(panel_errors) == 0 and mutation_ok,
    panel_detail,
    aid="triangle-shading.panel-window-set-exact")

  colour_errors = shading_colour_errors(axis_identities=axis_identities)
  colour_detail = "checked all z-order-zero collections and patches"
  if colour_errors:
    colour_detail += "; " + " | ".join(colour_errors)
  report(
    "every cut artist uses the shared rgba (_CUT_GREY)",
    len(colour_errors) == 0,
    colour_detail,
    aid="triangle-shading.all-cut-artists-use-shared-gray")

  marginal_errors = marginal_band_errors(
    axis_identities=axis_identities,
    cuts=cuts)
  marginal_detail = "lower and upper excluded intervals checked"
  if marginal_errors:
    marginal_detail += "; " + " | ".join(marginal_errors)
  report(
    "only the omegamh2 diagonal carries its two cut-bound bands",
    len(marginal_errors) == 0,
    marginal_detail,
    aid="triangle-shading.omegamh2-marginal-bands-exact")

  plt.close(figure)
  return finish()


if __name__ == "__main__":
  sys.exit(main())
