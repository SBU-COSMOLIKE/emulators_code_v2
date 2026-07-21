"""Imposed background physics: H(z) -> the cosmological distances.

The BSN design keeps the legacy emulbaosn insight: only H(z) is a
network; every distance is KNOWN physics computed from it. This module
owns that pipeline once, used by BOTH the cobaya adapter (emul_baosn)
and direct scripting off an EmulatorPredictor — the single-source rule,
so the integration convention can never fork between the two doors.

    H(z) on the stored grid (the emulator output, km/s/Mpc)
       │  c/H cubic-interpolated onto the DOUBLED grid
       │  z_step = linspace(0, z_max, 2*NZ + 1)   (odd point count)
       ▼
    chi(z) = integral_0^z c/H dz'      cumulative Simpson (composite
       │ even + one-interval odd)
       ▼
    flat conversions:  D_C = chi           comoving distance
                       D_A = chi / (1+z)   angular diameter
                       D_L = chi * (1+z)   luminosity

(legend: NZ = the stored grid's point count; z_max = its last redshift;
c = the speed of light in km/s, so every distance is in Mpc when H is
in km/s/Mpc.)

The pipeline is valid STRICTLY inside the grid window [0, z_max]. The
legacy analytic z->1200 extension (H_ext = H0*sqrt(om(1+z)^3 +
omegar(1+z)^4), self-labeled "this is an approximation") is NOT ported
(the two-regime design: the recombination window has its own trained
D_M emulator, so no bridging integration through the query desert
exists anywhere).
Curvature is V1 flat-only — the legacy curvature branch is dimensionally
wrong (sinh(chi*K)/K where the correct form needs sqrt(K); it agreed
with flat only because sinh(x) ~ x) and the corrected form is a recorded
future item behind a CAMB-comparison gate. The flat/desert guards live
in the adapter (it knows the query and the windows); this module is the
pure math.
"""

import numpy as np
from scipy import interpolate

# the speed of light in km/s — the legacy constant, verbatim, so chi is
# in Mpc when H is in km/s/Mpc.
C_KMS = 2.99792458e5


def cumulative_simpson(z, y):
  """Cumulative Simpson integral of y over the uniform grid z.

  Composite Simpson on each pair of intervals for the even points; for each
  odd point, the ONE-interval integral of the quadratic through the three
  surrounding samples, h/12 * (5*y[i-1] + 8*y[i] - y[i+1]). C[i] then
  approximates integral_{z[0]}^{z[i]} y dz at every grid point: exact on
  cubics at the even nodes and on quadratics at the odd nodes.

  REOPENED 2026-07-12: the original port computed the odd node as
  dz/6 * (y[i-1] + 4*y[i] + y[i+1]) -- HALF the two-interval Simpson total
  (the integral over the whole [z[i-1], z[i+1]] chunk), not the one-interval
  integral, a first-order h^2/2 error that the earlier "O(dz^3),
  bug-for-bug" acceptance mislabeled. It is superseded here by the correct
  one-interval rule; see families-background-mps.md.

  Arguments:
    z = (n,) UNIFORM ascending grid; n must be odd (an even number of
        intervals), n >= 3.
    y = (n,) integrand samples on z.

  Returns:
    (n,) the cumulative integral, C[0] = 0.

  Raises:
    ValueError on an even point count (Simpson needs interval pairs).
  """
  n = len(z)
  if n < 3 or (n - 1) % 2 != 0:
    raise ValueError("Need an odd number of points (even number of intervals).")
  dz = z[1] - z[0]

  # Simpson contributions on each pair of intervals [z[2m], z[2m+2]]
  # there are (n-1)/2 such chunks
  f0 = y[:-2:2]    # y[0], y[2], y[4], …
  f1 = y[1:-1:2]   # y[1], y[3], y[5], …
  f2 = y[2::2]     # y[2], y[4], y[6], …
  chunks = dz/3 * (f0 + 4*f1 + f2)

  # cumulative sum of the full-chunk integrals at even indices
  cum_even = np.concatenate(([0.0], np.cumsum(chunks)))

  # build the full cumulative array
  C = np.empty_like(y)
  C[0] = 0.0
  C[2::2] = cum_even[1:]               # at z[2], z[4], …
  # odd node i: the ONE-interval integral integral_{z[i-1]}^{z[i]} of the
  # quadratic through the three samples (y[i-1], y[i], y[i+1]) — exact on
  # quadratics. h/12 * (5*y[i-1] + 8*y[i] - y[i+1]). NOT dz/6 * (1,4,1),
  # which is HALF the two-interval Simpson total (integral over the whole
  # [z[i-1], z[i+1]] chunk) and a first-order h^2/2 error (reopen;
  # see families-background-mps.md).
  for i in range(1, n, 2):
    C[i] = C[i - 1] + dz / 12 * (5 * y[i - 1] + 8 * y[i] - y[i + 1])
  return C


def comoving_distance_grid(z_grid, h_grid):
  """chi(z) over the doubled grid from H(z) on the stored grid (flat).

  The legacy convention verbatim: c/H is cubic-interpolated onto the
  doubled uniform grid z_step = linspace(0, z_max, 2*NZ + 1) (odd point
  count by construction, starting at z = 0 so chi(0) = 0), then
  cumulative-Simpson integrated.

  Arguments:
    z_grid = (NZ,) the stored ascending redshift grid. It starts exactly at
             zero and z_grid[-1] is the window edge z_max.
    h_grid = (NZ,) H(z_grid) in km/s/Mpc.

  Returns:
    (z_step, chi): the doubled grid and the comoving distance on it,
    both (2*NZ + 1,), chi in Mpc.
  """
  z_grid = np.asarray(z_grid, dtype="float64")
  h_grid = np.asarray(h_grid, dtype="float64")
  grid_is_valid = (
    z_grid.ndim == 1 and len(z_grid) >= 4
    and np.isfinite(z_grid).all() and z_grid[0] == 0.0
    and (z_grid[1:] > z_grid[:-1]).all())
  if not grid_is_valid:
    raise ValueError(
      "the Hubble redshift grid must be finite, strictly increasing, "
      "one-dimensional, and start exactly at z = 0")
  if h_grid.shape != z_grid.shape or not np.isfinite(h_grid).all() \
      or not (h_grid > 0.0).all():
    raise ValueError(
      "H(z) must be one finite positive value for every redshift")

  func = interpolate.interp1d(z_grid, C_KMS / h_grid,
                              kind='cubic',
                              assume_sorted=True,
                              fill_value="extrapolate")
  zstep = np.linspace(0.0, z_grid[-1], 2 * len(z_grid) + 1)
  chi = cumulative_simpson(zstep, func(zstep))
  return zstep, chi


def distance_interpolators(z_grid, h_grid):
  """Build the SN-window background interpolators from one H(z) row.

  One call per sampled cosmology: integrates chi (comoving_distance_grid)
  and wraps H and the three flat distances in cubic interpolators over
  the window. Every consumer (the cobaya getters, a profile script, the
  diagnostics pages) reads these, so the convention exists once.

  Arguments:
    z_grid = (NZ,) the stored ascending redshift grid.
    h_grid = (NZ,) H(z_grid) in km/s/Mpc.

  Returns:
    a dict with:
      H     = cubic interpolator z -> H(z)  (km/s/Mpc)
      chi   = cubic interpolator z -> comoving distance D_C (Mpc)
      da    = cubic interpolator z -> angular-diameter distance (Mpc)
      dl    = cubic interpolator z -> luminosity distance (Mpc)
      z_max = the window edge (queries beyond it are the ADAPTER's
              loud-error job, not silent extrapolation).
  """
  zstep, chi = comoving_distance_grid(z_grid=z_grid, h_grid=h_grid)
  # flat conversions (legacy verbatim): dl = chi*(1+z), da = dl/(1+z)^2.
  dl = chi * (1.0 + zstep)
  da = dl / (1.0 + zstep) ** 2

  def cubic(x, y):
    """Build one cubic interpolant over the grid (extrapolation allowed).

    Arguments:
      x = the strictly increasing coordinate grid.
      y = the values on that grid.

    Returns:
      a scipy interp1d callable y(x).
    """
    return interpolate.interp1d(x, y,
                                kind='cubic',
                                assume_sorted=True,
                                fill_value="extrapolate")

  return {"H":     cubic(z_grid, np.asarray(h_grid)),
          "chi":   cubic(zstep, chi),
          "da":    cubic(zstep, da),
          "dl":    cubic(zstep, dl),
          "z_max": float(z_grid[-1])}
